//! Thread-local CUDA graph workspace routing.
//!
//! Selects which temporary scratch pool eager forwards, decode CUDA graphs, and
//! MTP/DFlash verify graphs use. This is quantization-agnostic: BF16, FP8, NVFP4,
//! MXFP4, and other CUDA paths share the same domain switch.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphWorkspaceDomain {
    Eager,
    DecodeGraph,
    MtpGraph,
}

thread_local! {
    static GRAPH_WORKSPACE_DOMAIN: std::cell::Cell<GraphWorkspaceDomain> =
        const { std::cell::Cell::new(GraphWorkspaceDomain::Eager) };
}

pub struct GraphWorkspaceGuard {
    previous: GraphWorkspaceDomain,
}

impl Drop for GraphWorkspaceGuard {
    fn drop(&mut self) {
        GRAPH_WORKSPACE_DOMAIN.with(|domain| domain.set(self.previous));
    }
}

/// Route subsequent CUDA scratch/workspace allocations to `domain`.
pub fn set_graph_workspace_domain(domain: GraphWorkspaceDomain) -> GraphWorkspaceGuard {
    let previous = GRAPH_WORKSPACE_DOMAIN.with(|current| {
        let previous = current.get();
        current.set(domain);
        previous
    });
    GraphWorkspaceGuard { previous }
}

#[cfg(feature = "cuda")]
pub(crate) fn graph_workspace_domain() -> GraphWorkspaceDomain {
    GRAPH_WORKSPACE_DOMAIN.with(|domain| domain.get())
}
