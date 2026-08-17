proc briskkv_report_power_hier {} {
  foreach required {DDC_FILE TARGET_LIBRARY REPORT_DIR TOP} {
    if {![info exists ::env($required)] ||
        [string trim $::env($required)] eq ""} {
      error "$required is required"
    }
  }

  set ddc_file [file normalize $::env(DDC_FILE)]
  set report_dir [file normalize $::env(REPORT_DIR)]
  file mkdir $report_dir

  if {![file exists $ddc_file]} {
    error "DDC_FILE does not exist: $ddc_file"
  }

  set target_libraries [split $::env(TARGET_LIBRARY) ":"]
  set_app_var target_library $target_libraries

  if {[info exists ::env(LINK_LIBRARY)] &&
      [string trim $::env(LINK_LIBRARY)] ne ""} {
    set_app_var link_library \
      [concat "*" [split $::env(LINK_LIBRARY) ":"]]
  } else {
    set_app_var link_library [concat "*" $target_libraries]
  }

  read_ddc $ddc_file
  current_design $::env(TOP)
  link

  check_design > "$report_dir/check_design.rpt"

  # 用来确认读取 DDC 后,总功耗是否仍与原报告一致.
  report_power > "$report_dir/power_check.rpt"

  # 本次真正需要的新报告.
  report_power -hierarchy > "$report_dir/power_hier.rpt"

  # 检查 DDC 是否保留了原有 activity annotation.
  if {[llength [info commands report_saif]] > 0} {
    report_saif -hierarchy > "$report_dir/saif_annotation_from_ddc.rpt"
  }
}

if {[catch {briskkv_report_power_hier} briskkv_error briskkv_options]} {
  puts stderr "BRISK-KV hierarchical power report FAILED: $briskkv_error"
  if {[dict exists $briskkv_options -errorinfo]} {
    puts stderr [dict get $briskkv_options -errorinfo]
  }
  exit 1
}

puts "BRISK-KV hierarchical power report completed successfully"
exit 0
