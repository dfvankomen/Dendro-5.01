# Idea — Iterator API + SFC-sort backing storage (deferred 2026-05-15)

## Context

After fix B (2026-05-15), EM4 graph long-haul has a 1.78× SFC residual on U_E2 (analytical zero). Bisection localized it to the post-AMR `buildGraphTwin + repartitionMeshGlobal` flow; wide cg-trace at step 31 showed ~1e-15 absolute diffs at 95% of cgs (ULP-floor). Writer-set evidence (`project_writer_set_refuted_2026-05-14`) says the iteration ORDER is bit-identical but `getElementNodalValues` reads different values at hanging nodes due to `E2N_CG` resolution differences.

So the primary attack path (audit hanging-node `E2N_CG` resolution) is targeted at the actual residual source. The idea below is a separate, longer-term code-quality investment that may or may not help — saved here so we can come back to it.

## The idea

Two coupled changes, sequenced:

### Phase 1 (pure refactor): `Mesh::localElements()` iterator API

Today, every consumer that wants to iterate LOCAL elements writes:

```cpp
for (unsigned int e = mesh->getElementLocalBegin();
     e < mesh->getElementLocalEnd(); e++) {
    // ... process m_uiAllElements[e]
}
```

This bakes in the assumption that LOCAL is a contiguous range in `m_uiAllElements`. Replace with:

```cpp
for (auto e : mesh->localElements()) {
    // ... process mesh->getAllElements()[e]
}
```

Initial implementation wraps the existing range — zero behavior change. Hand-rolled iterator class:

```cpp
class Mesh::LocalElementsRange {
    const Mesh* m_mesh;
public:
    struct iterator {
        const Mesh* m; uint32_t i;
        uint32_t operator*() const { return i; }
        iterator& operator++() { ++i; return *this; }
        bool operator!=(const iterator& o) const { return i != o.i; }
    };
    LocalElementsRange(const Mesh* m) : m_mesh(m) {}
    iterator begin() const { return {m_mesh, m_mesh->getElementLocalBegin()}; }
    iterator end()   const { return {m_mesh, m_mesh->getElementLocalEnd()}; }
};
inline LocalElementsRange Mesh::localElements() const {
    return LocalElementsRange(this);
}
```

Pure refactor. Land in one PR. Mechanical search-and-replace at call sites.

Also add `localBlocks()` for the same reason — block iteration is currently `for (auto& blk : mesh->getLocalBlockList())`.

### Phase 2 (experiment): SFC-sort `m_uiAllElements` on graph mesh

After repartition, sort `m_uiAllElements` in SFC order. LOCAL elements become interleaved with ghosts (no longer contiguous). Backing storage for "what's local" becomes:

```cpp
std::vector<uint32_t> m_uiLocalElementIds;  // indices into m_uiAllElements
```

`LocalElementsRange::begin/end` returns iterators into this vector. Callers don't notice — they still write `for (auto e : mesh->localElements())`.

Gate behind `DENDRO_SFC_SORT_ALL_ELEMENTS=1` initially for A/B testing.

## Why we're not doing this first

1. The 1.78× residual is at hanging-node `E2N_CG` resolution, not iteration order (writer-set is bit-identical per `project_writer_set_refuted_2026-05-14`).
2. Previous iteration-order experiments (`project_iter_order_exhausted_2026-05-13`) tested 4 different orderings and got at most 30% improvement.
3. The refactor is mechanical but touches every downstream consumer. Big diff, real review cost.

## Why it might still be worth doing eventually

- Cleaner API — call sites don't bake in assumptions about storage layout
- Easier to add future invariants (e.g., "iterate active blocks only", "iterate blocks of level N")
- Single change point — if storage changes, only the iterator changes
- Phase 2 may help with future variables where U_E2-like analytical-zero diagnostics matter more
- Even if Phase 2 doesn't close the gap, Phase 1 is strictly an improvement

## When to revisit

- If the hanging-node audit-extension attack doesn't close the gap and we still want to chase the residual
- If a future feature needs partition-invariant element ordering for other reasons (e.g., reproducibility of VTU output)
- If the codebase accumulates more places that bake in "LOCAL is contiguous in m_uiAllElements" — at which point a refactor pays back via avoided churn

## Estimated scope

- Phase 1 (refactor): ~30-50 call sites, ~1-2 days
- Phase 2 (sort + indirection): touches `repartitionMeshGlobal`, `buildE2NMap`, scatter map indexing. Risk of breaking unrelated invariants. ~3-5 days for implementation + A/B testing
