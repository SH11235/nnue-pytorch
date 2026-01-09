#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"
#include "../YaneuraOu/source/learn/learn.h"

#include <optional>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <functional>

#ifdef _WIN32
#include <ppl.h>
#else
#include <algorithm>
#include <execution>
#endif

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&packedSfenValues[0]), sizeof(Learner::PackedSfenValue) * n))
                {
                    vec.resize(n);
#ifdef _WIN32
                    concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
                        {
                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        });
#else
                    for (size_t i = 0; i < n; ++i)
                    {
                        vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                    }
#endif
                    return;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return;

                        continue;
                    }

                    m_eof = true;
                    return;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    struct BinSfenMultiFileInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenMultiFileInputStream(const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_filenames(filenames),
            m_currentFileIndex(0),
            m_eof(false),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
            if (!m_filenames.empty())
            {
                m_stream = std::fstream(m_filenames[0], openmode);
                m_eof = !m_stream;
            }
            else
            {
                m_eof = true;
            }
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool cycledOnce = false;

            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    // Current file exhausted, try next file
                    m_currentFileIndex++;
                    if (m_currentFileIndex >= m_filenames.size())
                    {
                        if (m_cyclic)
                        {
                            if (cycledOnce)
                            {
                                m_eof = true;
                                return std::nullopt;
                            }
                            m_currentFileIndex = 0;
                            cycledOnce = true;
                        }
                        else
                        {
                            m_eof = true;
                            return std::nullopt;
                        }
                    }

                    m_stream = std::fstream(m_filenames[m_currentFileIndex], openmode);
                    if (!m_stream)
                    {
                        m_eof = true;
                        return std::nullopt;
                    }
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            size_t totalRead = 0;
            bool cycledOnce = false;

            while (totalRead < n)
            {
                size_t remaining = n - totalRead;
                if (m_stream.read(reinterpret_cast<char*>(&packedSfenValues[totalRead]), sizeof(Learner::PackedSfenValue) * remaining))
                {
                    totalRead += remaining;
                }
                else
                {
                    // Read partial data if any
                    size_t partialRead = m_stream.gcount() / sizeof(Learner::PackedSfenValue);
                    totalRead += partialRead;

                    // Move to next file
                    m_currentFileIndex++;
                    if (m_currentFileIndex >= m_filenames.size())
                    {
                        if (m_cyclic)
                        {
                            if (cycledOnce)
                            {
                                break;
                            }
                            m_currentFileIndex = 0;
                            cycledOnce = true;
                        }
                        else
                        {
                            m_eof = true;
                            break;
                        }
                    }

                    m_stream = std::fstream(m_filenames[m_currentFileIndex], openmode);
                    if (!m_stream)
                    {
                        m_eof = true;
                        break;
                    }
                }
            }

            vec.resize(totalRead);
            for (size_t i = 0; i < totalRead; ++i)
            {
                vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenMultiFileInputStream() override {}

    private:
        std::vector<std::string> m_filenames;
        size_t m_currentFileIndex;
        std::fstream m_stream;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (filenames.empty())
            return nullptr;

        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filenames[0], BinSfenInputStream::extension))
        {
            if (filenames.size() == 1)
            {
                // Single file: use original implementation
                return std::make_unique<BinSfenInputStream>(filenames[0], cyclic, std::move(skipPredicate));
            }
            else
            {
                // Multiple files: use multi-file implementation
                return std::make_unique<BinSfenMultiFileInputStream>(filenames, cyclic, std::move(skipPredicate));
            }
        }

        return nullptr;
    }
}

#endif
