#pragma once
//
// Generic TOML parameter reader shared across generated solvers.
//
// Section-first lookup with fallbacks: [section].key (exact, then
// case-insensitive) -> flat top-level key -> "<legacy_prefix>key" ->
// "<alt_prefix>key". The two legacy prefixes (solver namespace / project
// prefix) are passed in by the caller; empty strings skip those fallbacks.
//
// Header-only; #include from a generated parameters.cpp (which links toml11).
// dendrolib itself has no toml dependency -- nothing in the library includes
// this header, only downstream solvers do (same pattern as bh_refine.h).
//
#include <toml.hpp>

#include <cctype>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace dendro_params {

// Case-insensitive key lookup within a TOML table. Returns the actual key
// present matching `key` ignoring ASCII case, or "" if none (lets the reader
// accept e.g. both par_p_plus and par_P_plus).
inline std::string find_key_ci(const toml::value& tbl, const std::string& key) {
    auto lower = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    if (!tbl.is_table()) return "";
    const std::string target = lower(key);
    for (const auto& kv : tbl.as_table()) {
        if (lower(kv.first) == target) return kv.first;
    }
    return "";
}

template <typename T>
bool try_read(const toml::value& root, const std::string& section,
              const std::string& key, T& out,
              const std::string& legacy_prefix = "",
              const std::string& alt_prefix    = "") {
    // 1) [section].key  (exact, then case-insensitive)
    if (root.contains(section)) {
        const auto& tbl = root.at(section);
        if (tbl.contains(key)) {
            out = toml::find<T>(tbl, key);
            return true;
        }
        std::string ci = find_key_ci(tbl, key);
        if (!ci.empty()) {
            out = toml::find<T>(tbl, ci);
            return true;
        }
    }
    // 2) flat top-level key
    if (root.contains(key)) {
        out = toml::find<T>(root, key);
        return true;
    }
    // 3) legacy "namespace::KEY"
    if (!legacy_prefix.empty()) {
        std::string legacy = legacy_prefix + key;
        if (root.contains(legacy)) {
            out = toml::find<T>(root, legacy);
            return true;
        }
    }
    // 4) project-prefixed "PREFIX_KEY"
    if (!alt_prefix.empty()) {
        std::string prefixed = alt_prefix + key;
        if (root.contains(prefixed)) {
            out = toml::find<T>(root, prefixed);
            return true;
        }
    }
    return false;
}

// Overload for fixed-size C arrays.
template <typename T, std::size_t N>
bool try_read_array(const toml::value& root, const std::string& section,
                    const std::string& key, T (&out)[N],
                    const std::string& legacy_prefix = "",
                    const std::string& alt_prefix    = "") {
    std::vector<T> vec;
    bool found = false;
    if (root.contains(section)) {
        const auto& tbl = root.at(section);
        if (tbl.contains(key)) {
            vec   = toml::find<std::vector<T>>(tbl, key);
            found = true;
        }
        if (!found) {
            std::string ci = find_key_ci(tbl, key);
            if (!ci.empty()) {
                vec   = toml::find<std::vector<T>>(tbl, ci);
                found = true;
            }
        }
    }
    if (!found && root.contains(key)) {
        vec   = toml::find<std::vector<T>>(root, key);
        found = true;
    }
    if (!found && !legacy_prefix.empty()) {
        std::string legacy = legacy_prefix + key;
        if (root.contains(legacy)) {
            vec   = toml::find<std::vector<T>>(root, legacy);
            found = true;
        }
    }
    if (!found && !alt_prefix.empty()) {
        std::string prefixed = alt_prefix + key;
        if (root.contains(prefixed)) {
            vec   = toml::find<std::vector<T>>(root, prefixed);
            found = true;
        }
    }
    if (!found) return false;

    // Short array returns false (not a partial copy): the caller's "using
    // default" path is the honest outcome, and it logs. Long array truncates.
    if (vec.size() != N) {
        std::cerr << "[param_reader] WARNING: array '" << key << "'";
        if (!section.empty()) std::cerr << " in section [" << section << "]";
        std::cerr << " has " << vec.size() << " element(s), expected " << N
                  << ". ";
        if (vec.size() < N) {
            std::cerr << "Keeping compiled defaults for ALL " << N
                      << " slots -- fix the par file." << std::endl;
            return false;
        }
        std::cerr << "Using the first " << N << ", ignoring the rest."
                  << std::endl;
    }

    for (std::size_t i = 0; i < N; ++i) out[i] = vec[i];
    return true;
}

}  // namespace dendro_params
