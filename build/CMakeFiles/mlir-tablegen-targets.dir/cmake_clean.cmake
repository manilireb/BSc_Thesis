file(REMOVE_RECURSE
  "Combine.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/mlir-tablegen-targets.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
