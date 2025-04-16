def _extract_last_analysis_justification(goal_data):
    """Extract the justification from the last analysis."""
    if "last_analysis" in goal_data and isinstance(goal_data["last_analysis"], dict):
        return goal_data["last_analysis"].get("justification", "")
    return ""

def _extract_last_analysis_strategic_guidance(goal_data):
    """Extract the strategic guidance from the last analysis."""
    if "last_analysis" in goal_data and isinstance(goal_data["last_analysis"], dict):
        return goal_data["last_analysis"].get("strategic_guidance", "No specific guidance provided.")
    return "No specific guidance provided." 