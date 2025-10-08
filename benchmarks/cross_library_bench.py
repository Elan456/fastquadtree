import system_info_collector
from quadtree_bench.main import main

main()
system_info = system_info_collector.collect_system_info()
print(system_info_collector.format_system_info_markdown_lite(system_info))
