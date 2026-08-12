# BRISK-KV logic-only synthesis template for Design Compiler 2018+.
# Required environment:
#   RTL_DIR, TARGET_LIBRARY
# Optional:
#   TOP, CLOCK_PERIOD, REPORT_DIR, LINK_LIBRARY

proc briskkv_dc_main {} {
  if {![info exists ::env(RTL_DIR)]} {
    error "RTL_DIR is required"
  }
  if {![info exists ::env(TARGET_LIBRARY)]} {
    error "TARGET_LIBRARY is required"
  }

  set rtl_dir [file normalize $::env(RTL_DIR)]
  set top [expr {[info exists ::env(TOP)] ? $::env(TOP) : "BriskKvAttentionTop"}]
  set clock_period [expr {
    [info exists ::env(CLOCK_PERIOD)] ? $::env(CLOCK_PERIOD) : 2.0
  }]
  set report_dir [file normalize [expr {
    [info exists ::env(REPORT_DIR)] ? $::env(REPORT_DIR) : "reports"
  }]]
  file mkdir $report_dir

  # A per-process WORK library prevents stale analyzed .db files from an older
  # run from replacing newly generated bodyless SRAM stubs.
  set work_dir [file join $report_dir "work_[pid]"]
  file mkdir $work_dir
  define_design_lib WORK -path $work_dir

  # Keep an ordinary Tcl variable as well as the DC application variable.
  # set_app_var does not create a procedure-local variable named
  # "target_library" in DC O-2018.
  set target_libraries [split $::env(TARGET_LIBRARY) ":"]
  set_app_var target_library $target_libraries
  if {[info exists ::env(LINK_LIBRARY)]} {
    set_app_var link_library [concat "*" [split $::env(LINK_LIBRARY) ":"]]
  } else {
    set_app_var link_library [concat "*" $target_libraries]
  }

  set memory_module_file [file join $rtl_dir memory_modules.tcl]
  if {![file exists $memory_module_file]} {
    error "Missing memory black-box list: $memory_module_file"
  }
  source $memory_module_file
  if {[llength $BRISKKV_MEMORY_MODULES] == 0} {
    error "BRISKKV_MEMORY_MODULES is empty"
  }

  set sv_files [glob -nocomplain -directory $rtl_dir *.sv]
  if {[llength $sv_files] == 0} {
    error "No SystemVerilog files found in $rtl_dir"
  }

  # DC O-2018 does not provide a set_black_box command. The generated stubs
  # carry the recognized syn_black_box attribute. Reject any behavioral array
  # or missing marker before DC sees the RTL.
  foreach memory_module $BRISKKV_MEMORY_MODULES {
    set stub_file [file join $rtl_dir "${memory_module}.sv"]
    if {![file exists $stub_file]} {
      error "Missing architectural memory stub: $stub_file"
    }
    set channel [open $stub_file r]
    set stub_text [read $channel]
    close $channel
    if {[string first "Memory\[" $stub_text] >= 0} {
      error "Behavioral memory implementation found in dc_logic stub: $stub_file"
    }
    if {[string first "syn_black_box" $stub_text] < 0} {
      error "Memory stub is missing syn_black_box marker: $stub_file"
    }
  }

  analyze -format sverilog $sv_files
  elaborate $top
  current_design $top
  link

  # Audit and preserve every elaborated SRAM instance. Matching zero is a
  # fatal export/hierarchy error, not a reason to continue with invalid PPA.
  set total_memory_instances 0
  set audit_file [file join $report_dir memory_blackboxes_precompile.rpt]
  set audit_channel [open $audit_file w]
  puts $audit_channel "BRISK-KV architectural SRAM black-box audit"
  foreach memory_module $BRISKKV_MEMORY_MODULES {
    set memory_design [get_designs -quiet $memory_module]
    if {[sizeof_collection $memory_design] == 0} {
      close $audit_channel
      error "Architectural memory design not found after elaborate: $memory_module"
    }

    set memory_cells [get_cells -quiet -hierarchical -filter "ref_name == $memory_module"]
    set instance_count [sizeof_collection $memory_cells]
    if {$instance_count == 0} {
      close $audit_channel
      error "No elaborated architectural memory instance found for: $memory_module"
    }

    # The source attribute establishes black-box status. These constraints
    # prevent compile_ultra from absorbing or ungrouping the interface cell.
    set_dont_touch $memory_cells true
    set_ungroup $memory_cells false
    set total_memory_instances [expr {$total_memory_instances + $instance_count}]
    puts $audit_channel "$memory_module instances=$instance_count source=syn_black_box"
  }
  puts $audit_channel "total_instances=$total_memory_instances"
  close $audit_channel

  create_clock -name clk -period $clock_period [get_ports clock]
  set_clock_uncertainty [expr {$clock_period * 0.05}] [get_clocks clk]
  set_input_delay [expr {$clock_period * 0.10}] -clock clk \
    [remove_from_collection [all_inputs] [get_ports clock]]
  set_output_delay [expr {$clock_period * 0.10}] -clock clk [all_outputs]
  set_false_path -from [get_ports reset]

  check_design > "$report_dir/check_design.rpt"
  check_timing > "$report_dir/check_timing.rpt"
  compile_ultra

  # The dont_touch instances must still be present after compile.
  set post_audit_file [file join $report_dir memory_blackboxes_postcompile.rpt]
  set post_audit_channel [open $post_audit_file w]
  puts $post_audit_channel "BRISK-KV post-compile SRAM black-box audit"
  foreach memory_module $BRISKKV_MEMORY_MODULES {
    set memory_cells [get_cells -quiet -hierarchical -filter "ref_name == $memory_module"]
    set instance_count [sizeof_collection $memory_cells]
    if {$instance_count == 0} {
      close $post_audit_channel
      error "Architectural memory instance disappeared during compile: $memory_module"
    }
    puts $post_audit_channel "$memory_module instances=$instance_count"
  }
  close $post_audit_channel

  report_units > "$report_dir/units.rpt"
  report_clock > "$report_dir/clocks.rpt"
  report_qor > "$report_dir/qor.rpt"
  report_area -hierarchy > "$report_dir/area_hier.rpt"
  report_timing -delay_type max -max_paths 20 -transition_time -nets \
    > "$report_dir/timing_setup.rpt"
  report_timing -delay_type min -max_paths 20 -transition_time -nets \
    > "$report_dir/timing_hold.rpt"
  report_constraint -all_violators > "$report_dir/constraints.rpt"
  report_power > "$report_dir/power.rpt"
  report_reference -hierarchy > "$report_dir/references.rpt"

  write -format ddc -hierarchy -output "$report_dir/${top}.ddc"
  write -format verilog -hierarchy -output "$report_dir/${top}_mapped.v"
}

# dc_shell may otherwise continue after a command error in a -f script. Catch
# the complete flow and return an explicit process status to shell/tmux jobs.
if {[catch {briskkv_dc_main} briskkv_error briskkv_options]} {
  puts stderr "BRISK-KV DC FAILED: $briskkv_error"
  if {[dict exists $briskkv_options -errorinfo]} {
    puts stderr [dict get $briskkv_options -errorinfo]
  }
  exit 1
}

puts "BRISK-KV DC completed successfully"
exit 0
