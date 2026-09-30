/**
 * @file cceH5Dat.h
 * @brief Minimal HDF5 writer producing "h5::Dat"-compatible time-series
 *        datasets, for writing a Cauchy-Characteristic Extraction (CCE)
 *        worldtube file consumable by SpECTRE's PreprocessCceWorldtube.
 *
 * This does NOT link against SpECTRE's C++ source (a separate codebase/
 * build) -- it is a small, self-contained reimplementation of the on-disk
 * byte layout that SpECTRE's h5::Dat class (src/IO/H5/Dat.cpp in the
 * spectre repo) produces, using the raw HDF5 C API directly. Confirmed by
 * direct reading of h5::Dat's source (see Dendro_CCE_v2.0.md Section 6 for
 * citations):
 *   - each named quantity is stored as an extensible 2D double dataset
 *     named "<Name>.dat" (e.g. "/Lapse.dat"), initial size {0, ncols},
 *     chunked at {4, ncols}, max size {H5S_UNLIMITED, ncols}, with
 *     H5D_FILL_TIME_NEVER set;
 *   - a scalar uint32 attribute "version.ver" (SpECTRE's default is 1);
 *   - a rank-1 string attribute "Legend" of length ncols, stored with HDF5
 *     type H5T_FORTRAN_S1 (variable-length), written from a C-string-array
 *     buffer of type H5T_C_S1 (variable-length, ASCII, null-terminated).
 *   - SpECTRE's h5::Dat also optionally writes a "header.hdr" attribute
 *     (build/environment info via their "Formaline" system) -- confirmed
 *     from source that this is read only if H5Aexists() finds it, i.e. it
 *     is NOT required for a file to be read successfully, so this writer
 *     omits it.
 * This has NOT been verified against a real PreprocessCceWorldtube run --
 * see Dendro_CCE_v2.0.md Section 7/8 for the recommended validation step
 * before trusting this in production.
 */

#ifndef DENDRO_GR_CCE_H5DAT_H
#define DENDRO_GR_CCE_H5DAT_H

#include <hdf5.h>

#include <string>
#include <vector>

namespace cce_io {

/**
 * @brief Owns one open HDF5 file and a set of h5::Dat-style extensible
 *        time-series datasets inside it (one per worldtube quantity, e.g.
 *        "gxx", "Lapse", "AuxiliaryShiftx", ...).
 */
class H5DatFile {
   public:
    /**@brief Creates a new HDF5 file at `path` (overwrites if it exists).*/
    explicit H5DatFile(const std::string& path);

    ~H5DatFile();

    H5DatFile(const H5DatFile&)            = delete;
    H5DatFile& operator=(const H5DatFile&) = delete;

    /**
     * @brief Creates a new extensible dataset named `name + ".dat"` with
     *        `legend.size()` columns (must equal 1 + number_of_points, i.e.
     *        the time column plus one column per SWSH collocation point).
     *        Must be called once per quantity before any append_row() call
     *        for that quantity.
     */
    void create_dataset(const std::string& name,
                        const std::vector<std::string>& legend);

    /**
     * @brief Appends one row (e.g. one timestep) to the named dataset.
     *        `row.size()` must equal the number of columns the dataset was
     *        created with (including the leading time column).
     */
    void append_row(const std::string& name, const std::vector<double>& row);

    /**
     * @brief Flushes all pending writes to disk, leaving the file in a
     * valid, independently-readable state as of this call. `cce_file` in
     * BSSNCtx::writeCceWorldtube() is a `static` raw pointer that is never
     * explicitly `delete`d, so its destructor (which calls H5Fclose) never
     * runs on either a normal or killed process exit -- call this
     * periodically (writeCceWorldtube() calls it once per timestep, after
     * writing all 49 datasets' rows) so the file is always safely readable
     * regardless of how/when the run ends, rather than relying on a clean
     * close that never actually happens.
     */
    void flush();

   private:
    struct DatasetHandle {
        hid_t dataset_id;
        hsize_t num_rows;
        hsize_t num_cols;
    };

    hid_t file_id_;
    std::vector<std::pair<std::string, DatasetHandle>> datasets_;

    DatasetHandle& find_dataset(const std::string& name);
};

}  // namespace cce_io

#endif  // DENDRO_GR_CCE_H5DAT_H
