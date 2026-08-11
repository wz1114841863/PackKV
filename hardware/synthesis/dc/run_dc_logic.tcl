# BRISK-KV logic-only synthesis template.
# Required environment:
#   RTL_DIR, TARGET_LIBRARY
# Optional:
#   TOP, CLOCK_PERIOD, REPORT_DIR, LINK_LIBRARY

if {![info exists ::env(RTL_DIR)]} {
  error "RTL_DIR is required"
}
if {![info exists ::env(TARGET_LIBRARY)]} {
  error "TARGET_LIBRARY is required"
}

set rtl_dir [file normalize $::env(RTL_DIR)]
set top [expr {[info exists ::env(TOP)] ? $::env(TOP) : "BriskKvAttentionTop"}]
set clock_period [expr {[info exists ::env(CLOCK_PERIOD)] ? $::env(CLOCK_PERIOD) : 2.0}]
set report_dir [expr {[info exists ::env(REPORT_DIR)] ? $::env(REPORT_DIR) : "reports"}]
file mkdir $report_dir

set_app_var target_library [split $::env(TARGET_LIBRARY) ":"]
if {[info exists ::env(LINK_LIBRARY)]} {
  set_app_var link_library [concat "*" [split $::env(LINK_LIBRARY) ":"]]
} else {
  set_app_var link_library [concat "*" $target_library]
}

set sv_files [glob -nocomplain -directory $rtl_dir *.sv]
if {[llength $sv_files] == 0} {
  error "No SystemVerilog files found in $rtl_dir"
}

analyze -format sverilog $sv_files

set memory_module_file [file join $rtl_dir memory_modules.tcl]
if {![file exists $memory_module_file]} {
  error "Missing memory black-box list: $memory_module_file"
}
source $memory_module_file
foreach memory_module $BRISKKV_MEMORY_MODULES {
  set memory_design [get_designs -quiet $memory_module]
  if {[sizeof_collection $memory_design] == 0} {
    error "Architectural memory design not found: $memory_module"
  }
  set_black_box $memory_design
}

elaborate $top
current_design $top
link

create_clock -name clk -period $clock_period [get_ports clock]
set_clock_uncertainty [expr {$clock_period * 0.05}] [get_clocks clk]
set_input_delay [expr {$clock_period * 0.10}] -clock clk \
  [remove_from_collection [all_inputs] [get_ports clock]]
set_output_delay [expr {$clock_period * 0.10}] -clock clk [all_outputs]
set_false_path -from [get_ports reset]

check_design > "$report_dir/check_design.rpt"
compile_ultra

report_qor > "$report_dir/qor.rpt"
report_area -hierarchy > "$report_dir/area_hier.rpt"
report_timing -max_paths 20 -transition_time -nets > "$report_dir/timing.rpt"
report_power > "$report_dir/power.rpt"
report_reference -hierarchy > "$report_dir/references.rpt"

write -format ddc -hierarchy -output "$report_dir/${top}.ddc"
write -format verilog -hierarchy -output "$report_dir/${top}_mapped.v"
