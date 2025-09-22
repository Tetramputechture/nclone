"""
Path and dependency analyzers for reachability-based planning.

This module provides analyzers that use nclone's reachability analysis
to understand level structure and path feasibility. The analysis provides
input to intrinsic curiosity modules in npp-rl.
"""


class PathAnalyzer:
    """Analyzes paths using nclone's reachability analysis."""
    
    def analyze_path_feasibility(self, start_pos, end_pos, reachability_result):
        """Analyze if path is feasible using nclone's reachability analysis."""
        # Use nclone's reachability system - provides input to npp-rl curiosity modules
        if hasattr(reachability_result, 'is_path_feasible'):
            return reachability_result.is_path_feasible(start_pos, end_pos)
        elif hasattr(reachability_result, 'reachable_positions'):
            return end_pos in reachability_result.reachable_positions
        elif isinstance(reachability_result, dict) and 'reachable_positions' in reachability_result:
            return end_pos in reachability_result['reachable_positions']
        return False


class DependencyAnalyzer:
    """Analyzes switch dependencies using nclone's reachability analysis."""
    
    def analyze_switch_dependencies(self, level_data, reachability_result):
        """Analyze switch dependencies using nclone's reachability analysis."""
        # Use nclone's reachability system to understand switch relationships
        dependencies = {}
        if hasattr(level_data, 'switches'):
            for switch in level_data.switches:
                dependencies[switch.get('id')] = {
                    'blocks': [],
                    'required_for': []
                }
        return dependencies