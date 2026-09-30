#include "cceH5Dat.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace cce_io {

namespace {

void check_h5(const herr_t status, const std::string& msg) {
    if (status < 0) {
        throw std::runtime_error("cce_io::H5DatFile: " + msg);
    }
}

void check_h5_id(const hid_t id, const std::string& msg) {
    if (id < 0) {
        throw std::runtime_error("cce_io::H5DatFile: " + msg);
    }
}

// Writes the scalar "version.ver" attribute on `dataset_id`, matching
// h5::Version's on-disk format (a scalar H5T_NATIVE_UINT32 attribute).
void write_version_attribute(const hid_t dataset_id, const uint32_t version) {
    const hid_t space_id = H5Screate(H5S_SCALAR);
    check_h5_id(space_id, "failed to create scalar dataspace for version");
    const hid_t attr_id =
        H5Acreate2(dataset_id, "version.ver", H5T_NATIVE_UINT32, space_id,
                   H5P_DEFAULT, H5P_DEFAULT);
    check_h5_id(attr_id, "failed to create version.ver attribute");
    check_h5(H5Awrite(attr_id, H5T_NATIVE_UINT32, &version),
            "failed to write version.ver attribute");
    check_h5(H5Aclose(attr_id), "failed to close version.ver attribute");
    check_h5(H5Sclose(space_id), "failed to close version.ver dataspace");
}

// Writes the "Legend" attribute, matching h5::Dat's on-disk format: stored
// attribute type is a variable-length H5T_FORTRAN_S1 string, written from a
// memory buffer of variable-length H5T_C_S1 (ASCII, null-terminated)
// C-string pointers. See spectre/src/IO/H5/Helpers.cpp,
// write_to_attribute<std::string>(...) (specialization for
// std::vector<std::string>), which this mirrors.
void write_legend_attribute(const hid_t dataset_id,
                            const std::vector<std::string>& legend) {
    hid_t stored_type_id = H5Tcopy(H5T_FORTRAN_S1);
    check_h5_id(stored_type_id, "failed to copy H5T_FORTRAN_S1");
    check_h5(H5Tset_size(stored_type_id, H5T_VARIABLE),
            "failed to set Legend stored type to variable length");

    const hsize_t dim     = legend.size();
    const hid_t space_id  = H5Screate_simple(1, &dim, nullptr);
    check_h5_id(space_id, "failed to create Legend dataspace");
    const hid_t attr_id = H5Acreate2(dataset_id, "Legend", stored_type_id,
                                     space_id, H5P_DEFAULT, H5P_DEFAULT);
    check_h5_id(attr_id, "failed to create Legend attribute");

    hid_t mem_type_id = H5Tcopy(H5T_C_S1);
    check_h5_id(mem_type_id, "failed to copy H5T_C_S1");
    check_h5(H5Tset_size(mem_type_id, H5T_VARIABLE),
            "failed to set Legend mem type to variable length");
    check_h5(H5Tset_cset(mem_type_id, H5T_CSET_ASCII),
            "failed to set Legend mem type to ASCII");
    check_h5(H5Tset_strpad(mem_type_id, H5T_STR_NULLTERM),
            "failed to set Legend mem type to null-terminated");

    std::vector<const char*> pointers(legend.size());
    std::transform(legend.begin(), legend.end(), pointers.begin(),
                   [](const std::string& s) { return s.c_str(); });
    check_h5(H5Awrite(attr_id, mem_type_id, pointers.data()),
            "failed to write Legend attribute");

    check_h5(H5Aclose(attr_id), "failed to close Legend attribute");
    check_h5(H5Sclose(space_id), "failed to close Legend dataspace");
    check_h5(H5Tclose(mem_type_id), "failed to close Legend mem type");
    check_h5(H5Tclose(stored_type_id), "failed to close Legend stored type");
}

}  // namespace

H5DatFile::H5DatFile(const std::string& path) {
    file_id_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                         H5P_DEFAULT);
    check_h5_id(file_id_, "failed to create file '" + path + "'");
}

H5DatFile::~H5DatFile() {
    for (auto& entry : datasets_) {
        H5Dclose(entry.second.dataset_id);
    }
    H5Fclose(file_id_);
}

void H5DatFile::create_dataset(const std::string& name,
                               const std::vector<std::string>& legend) {
    const std::string dataset_name = name + ".dat";
    const hsize_t ncols            = legend.size();

    // Matches h5::detail::create_extensible_dataset: initial size {0,
    // ncols}, chunked at {4, ncols} (SpECTRE's own default chunk size for
    // Dat objects), max size {H5S_UNLIMITED, ncols}, fill time never (we
    // always write full rows via append_row, so there's nothing to
    // pre-fill).
    const hsize_t initial_size[2] = {0, ncols};
    const hsize_t chunk_size[2]   = {4, ncols};
    const hsize_t max_size[2]     = {H5S_UNLIMITED, ncols};

    const hid_t dataspace_id =
        H5Screate_simple(2, initial_size, max_size);
    check_h5_id(dataspace_id, "failed to create dataspace for '" +
                                  dataset_name + "'");

    const hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    check_h5_id(plist_id, "failed to create property list for '" +
                              dataset_name + "'");
    check_h5(H5Pset_chunk(plist_id, 2, chunk_size),
            "failed to set chunk size for '" + dataset_name + "'");
    check_h5(H5Pset_fill_time(plist_id, H5D_FILL_TIME_NEVER),
            "failed to set fill time for '" + dataset_name + "'");

    const hid_t dataset_id =
        H5Dcreate2(file_id_, dataset_name.c_str(), H5T_NATIVE_DOUBLE,
                  dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    check_h5_id(dataset_id, "failed to create dataset '" + dataset_name +
                                "'");

    check_h5(H5Pclose(plist_id), "failed to close property list for '" +
                                     dataset_name + "'");
    check_h5(H5Sclose(dataspace_id), "failed to close dataspace for '" +
                                         dataset_name + "'");

    write_version_attribute(dataset_id, 1);
    write_legend_attribute(dataset_id, legend);

    datasets_.push_back({name, DatasetHandle{dataset_id, 0, ncols}});
}

H5DatFile::DatasetHandle& H5DatFile::find_dataset(const std::string& name) {
    for (auto& entry : datasets_) {
        if (entry.first == name) {
            return entry.second;
        }
    }
    throw std::runtime_error(
        "cce_io::H5DatFile: append_row() called on unknown dataset '" +
        name + "' -- call create_dataset() first");
}

void H5DatFile::append_row(const std::string& name,
                           const std::vector<double>& row) {
    DatasetHandle& handle = find_dataset(name);
    if (row.size() != handle.num_cols) {
        throw std::runtime_error(
            "cce_io::H5DatFile: append_row() for '" + name +
            "' received a row of " + std::to_string(row.size()) +
            " entries, but the dataset has " +
            std::to_string(handle.num_cols) + " columns");
    }

    const hsize_t new_num_rows       = handle.num_rows + 1;
    const hsize_t new_size[2]        = {new_num_rows, handle.num_cols};
    check_h5(H5Dset_extent(handle.dataset_id, new_size),
            "failed to extend dataset '" + name + "'");

    const hid_t filespace_id = H5Dget_space(handle.dataset_id);
    check_h5_id(filespace_id, "failed to get dataspace when appending to '" +
                                  name + "'");
    const hsize_t start[2] = {handle.num_rows, 0};
    const hsize_t count[2] = {1, handle.num_cols};
    check_h5(H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start,
                                 nullptr, count, nullptr),
            "failed to select hyperslab when appending to '" + name + "'");

    const hid_t memspace_id = H5Screate_simple(2, count, nullptr);
    check_h5_id(memspace_id, "failed to create memory dataspace when "
                             "appending to '" +
                                 name + "'");

    check_h5(H5Dwrite(handle.dataset_id, H5T_NATIVE_DOUBLE, memspace_id,
                      filespace_id, H5P_DEFAULT, row.data()),
            "failed to write row when appending to '" + name + "'");

    check_h5(H5Sclose(memspace_id), "failed to close memory dataspace when "
                                     "appending to '" +
                                         name + "'");
    check_h5(H5Sclose(filespace_id), "failed to close file dataspace when "
                                      "appending to '" +
                                          name + "'");

    handle.num_rows = new_num_rows;
}

void H5DatFile::flush() {
    check_h5(H5Fflush(file_id_, H5F_SCOPE_GLOBAL),
            "failed to flush file");
}

}  // namespace cce_io
