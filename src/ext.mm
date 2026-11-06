// PyBind11 bindings for MtlMesh
#import <torch/extension.h>
#import "mtlmesh.h"
#import "metal_hash.h"

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

// Drains everything a call autoreleases (Metal command buffers, encoders, buffers returned at +0) when it
// returns. Python never drains an autorelease pool, so without this they live until the process exits.
struct AutoreleasePool {
    void* pool = objc_autoreleasePoolPush();
    ~AutoreleasePool() { objc_autoreleasePoolPop(pool); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<mtlmesh::MtlMesh>(m, "MtlMesh")
        .def(py::init<>())
        .def("num_vertices", &mtlmesh::MtlMesh::num_vertices, py::call_guard<AutoreleasePool>())
        .def("num_faces", &mtlmesh::MtlMesh::num_faces, py::call_guard<AutoreleasePool>())
        .def("num_edges", &mtlmesh::MtlMesh::num_edges, py::call_guard<AutoreleasePool>())
        .def("num_boundaries", &mtlmesh::MtlMesh::num_boundaries, py::call_guard<AutoreleasePool>())
        .def("num_conneted_components", &mtlmesh::MtlMesh::num_conneted_components, py::call_guard<AutoreleasePool>())
        .def("num_boundary_conneted_components", &mtlmesh::MtlMesh::num_boundary_conneted_components, py::call_guard<AutoreleasePool>())
        .def("num_boundary_loops", &mtlmesh::MtlMesh::num_boundary_loops, py::call_guard<AutoreleasePool>())
        .def("clear_cache", &mtlmesh::MtlMesh::clear_cache, py::call_guard<AutoreleasePool>())
        .def("init", &mtlmesh::MtlMesh::init, py::call_guard<AutoreleasePool>())
        .def("read", &mtlmesh::MtlMesh::read, py::call_guard<AutoreleasePool>())
        .def("read_face_areas", &mtlmesh::MtlMesh::read_face_areas, py::call_guard<AutoreleasePool>())
        .def("read_face_normals", &mtlmesh::MtlMesh::read_face_normals, py::call_guard<AutoreleasePool>())
        .def("read_vertex_normals", &mtlmesh::MtlMesh::read_vertex_normals, py::call_guard<AutoreleasePool>())
        .def("read_edges", &mtlmesh::MtlMesh::read_edges, py::call_guard<AutoreleasePool>())
        .def("read_boundaries", &mtlmesh::MtlMesh::read_boundaries, py::call_guard<AutoreleasePool>())
        .def("read_manifold_face_adjacency", &mtlmesh::MtlMesh::read_manifold_face_adjacency, py::call_guard<AutoreleasePool>())
        .def("read_manifold_boundary_adjacency", &mtlmesh::MtlMesh::read_manifold_boundary_adjacency, py::call_guard<AutoreleasePool>())
        .def("read_connected_components", &mtlmesh::MtlMesh::read_connected_components, py::call_guard<AutoreleasePool>())
        .def("read_boundary_connected_components", &mtlmesh::MtlMesh::read_boundary_connected_components, py::call_guard<AutoreleasePool>())
        .def("read_boundary_loops", &mtlmesh::MtlMesh::read_boundary_loops, py::call_guard<AutoreleasePool>())
        .def("read_all_cache", &mtlmesh::MtlMesh::read_all_cache, py::call_guard<AutoreleasePool>())
        .def("compute_face_areas", &mtlmesh::MtlMesh::compute_face_areas, py::call_guard<AutoreleasePool>())
        .def("compute_face_normals", &mtlmesh::MtlMesh::compute_face_normals, py::call_guard<AutoreleasePool>())
        .def("compute_vertex_normals", &mtlmesh::MtlMesh::compute_vertex_normals, py::call_guard<AutoreleasePool>())
        .def("get_vertex_face_adjacency", &mtlmesh::MtlMesh::get_vertex_face_adjacency, py::call_guard<AutoreleasePool>())
        .def("get_edges", &mtlmesh::MtlMesh::get_edges, py::call_guard<AutoreleasePool>())
        .def("get_edge_face_adjacency", &mtlmesh::MtlMesh::get_edge_face_adjacency, py::call_guard<AutoreleasePool>())
        .def("get_vertex_edge_adjacency", &mtlmesh::MtlMesh::get_vertex_edge_adjacency, py::call_guard<AutoreleasePool>())
        .def("get_boundary_info", &mtlmesh::MtlMesh::get_boundary_info, py::call_guard<AutoreleasePool>())
        .def("get_vertex_boundary_adjacency", &mtlmesh::MtlMesh::get_vertex_boundary_adjacency, py::call_guard<AutoreleasePool>())
        .def("get_vertex_is_manifold", &mtlmesh::MtlMesh::get_vertex_is_manifold, py::call_guard<AutoreleasePool>())
        .def("get_manifold_face_adjacency", &mtlmesh::MtlMesh::get_manifold_face_adjacency, py::call_guard<AutoreleasePool>())
        .def("get_manifold_boundary_adjacency", &mtlmesh::MtlMesh::get_manifold_boundary_adjacency, py::call_guard<AutoreleasePool>())
        .def("get_connected_components", &mtlmesh::MtlMesh::get_connected_components, py::call_guard<AutoreleasePool>())
        .def("get_boundary_connected_components", &mtlmesh::MtlMesh::get_boundary_connected_components, py::call_guard<AutoreleasePool>())
        .def("get_boundary_loops", &mtlmesh::MtlMesh::get_boundary_loops, py::call_guard<AutoreleasePool>())
        .def("remove_faces", &mtlmesh::MtlMesh::remove_faces, py::call_guard<AutoreleasePool>())
        .def("remove_unreferenced_vertices", &mtlmesh::MtlMesh::remove_unreferenced_vertices, py::call_guard<AutoreleasePool>())
        .def("remove_duplicate_faces", &mtlmesh::MtlMesh::remove_duplicate_faces, py::call_guard<AutoreleasePool>())
        .def("remove_degenerate_faces", &mtlmesh::MtlMesh::remove_degenerate_faces, py::call_guard<AutoreleasePool>())
        .def("fill_holes", &mtlmesh::MtlMesh::fill_holes, py::call_guard<AutoreleasePool>())
        .def("repair_non_manifold_edges", &mtlmesh::MtlMesh::repair_non_manifold_edges, py::call_guard<AutoreleasePool>())
        .def("remove_non_manifold_faces", &mtlmesh::MtlMesh::remove_non_manifold_faces, py::call_guard<AutoreleasePool>())
        .def("remove_small_connected_components", &mtlmesh::MtlMesh::remove_small_connected_components, py::call_guard<AutoreleasePool>())
        .def("unify_face_orientations", &mtlmesh::MtlMesh::unify_face_orientations, py::call_guard<AutoreleasePool>())
        .def("simplify_step", &mtlmesh::MtlMesh::simplify_step,
             py::arg("lambda_edge_length"), py::arg("lambda_skinny"),
             py::arg("threshold"), py::arg("timing") = false,
             py::call_guard<AutoreleasePool>())
        .def("compute_charts", &mtlmesh::MtlMesh::compute_charts, py::call_guard<AutoreleasePool>())
        .def("read_atlas_charts", &mtlmesh::MtlMesh::read_atlas_charts, py::call_guard<AutoreleasePool>());

    // Hash table functions (for remeshing)
    m.def("hashmap_insert_3d_idx_as_val", &mtlmesh::hashmap_insert_3d_idx_as_val,
          py::arg("keys"), py::arg("vals"), py::arg("coords"),
          py::arg("W"), py::arg("H"), py::arg("D"),
          "Insert 3D coords into spatial hash table with index-as-value",
             py::call_guard<AutoreleasePool>());

    m.def("hashmap_lookup_3d", &mtlmesh::hashmap_lookup_3d,
          py::arg("keys"), py::arg("vals"), py::arg("coords"),
          py::arg("W"), py::arg("H"), py::arg("D"),
          "Lookup 3D coords in spatial hash table",
             py::call_guard<AutoreleasePool>());

    m.def("get_sparse_voxel_grid_active_vertices", &mtlmesh::get_sparse_voxel_grid_active_vertices,
          py::arg("keys"), py::arg("vals"), py::arg("coords"),
          py::arg("W"), py::arg("H"), py::arg("D"),
          "Get active vertices for sparse voxel grid",
             py::call_guard<AutoreleasePool>());

    m.def("simple_dual_contour", &mtlmesh::simple_dual_contour,
          py::arg("keys"), py::arg("vals"), py::arg("coords"), py::arg("distances"),
          py::arg("W"), py::arg("H"), py::arg("D"),
          "Simple dual contouring on sparse voxel grid",
             py::call_guard<AutoreleasePool>());
}
