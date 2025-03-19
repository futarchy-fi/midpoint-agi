import asyncio
from typing import List, Dict, Any
import re
import os
import random

from .models import Goal, ExecutionResult, ValidationResult
from .tools import (
    get_current_hash,
    validate_repository_state,
    list_directory,
    read_file,
    search_code,
    run_terminal_cmd,
    get_current_branch,
    web_search,
    web_scrape
)

class GoalValidator:
    """Agent responsible for validating execution results against goals."""
    
    def __init__(self):
        self.system_prompt = """You are a goal validation agent responsible for evaluating execution results.
Your role is to:
1. Check if execution was successful
2. Validate changes against goal criteria
3. Provide detailed reasoning for validation decisions
4. Calculate a validation score

Available tools:
- list_directory: List contents of a directory
- read_file: Read contents of a file
- search_code: Search for code patterns
- run_terminal_cmd: Run a terminal command
- web_search: Search the web using DuckDuckGo's API
- web_scrape: Scrape content from a webpage

Always validate against the specific criteria provided in the goal.
Provide clear reasoning for your validation decisions."""

    async def validate_execution(self, goal: Goal, execution_result: ExecutionResult) -> ValidationResult:
        """
        Validate an execution result against a goal.
        
        This is an intelligent agent that:
        1. Understands validation criteria
        2. Uses repository exploration tools to collect evidence
        3. Makes judgments about whether criteria are satisfied
        4. Provides detailed reasoning for its decisions
        
        Args:
            goal: The goal to validate against
            execution_result: The result of task execution
            
        Returns:
            ValidationResult containing the validation outcome
        """
        # If execution failed, validation fails
        if not execution_result.success:
            return ValidationResult(
                success=False,
                score=0.0,
                reasoning="Execution failed: " + (execution_result.error_message or "Unknown error"),
                criteria_results=[],
                git_hash=execution_result.git_hash,
                branch_name=execution_result.branch_name
            )
        
        # Validate repository state
        try:
            await validate_repository_state(
                execution_result.repository_path,
                execution_result.git_hash
            )
        except ValueError as e:
            # Repository state validation can fail if we're not on the right branch
            # Let's try to check out the branch first and then validate
            pass
        
        # Check which branch we're on
        current_branch = await get_current_branch(execution_result.repository_path)
        
        # Switch to the execution branch if needed
        if current_branch != execution_result.branch_name:
            try:
                await run_terminal_cmd(
                    command=["git", "checkout", execution_result.branch_name],
                    cwd=execution_result.repository_path
                )
            except Exception as e:
                return ValidationResult(
                    success=False,
                    score=0.0,
                    reasoning=f"Failed to checkout branch {execution_result.branch_name}: {str(e)}",
                    criteria_results=[],
                    git_hash=execution_result.git_hash,
                    branch_name=execution_result.branch_name
                )
        
        try:
            # Explore repository state after execution
            repo_contents = await list_directory(execution_result.repository_path)
            
            # Evaluate each criterion
            criteria_results = []
            total_score = 0.0
            
            for criterion in goal.validation_criteria:
                # Initialize validation for this criterion
                criterion_result = {
                    "criterion": criterion,
                    "passed": False,
                    "reasoning": "",
                    "evidence": []
                }
                
                # Analyze criterion to determine validation approach
                if "file" in criterion.lower():
                    # Extract file names mentioned in criterion
                    file_names = re.findall(r'["\']([^"\']+)["\']', criterion)
                    if not file_names:
                        # Try to extract file names without quotes
                        file_names = re.findall(r'\b(\w+\.\w+)\b', criterion)
                    
                    # Check if files exist
                    for file_name in file_names:
                        try:
                            # Read the file to verify it exists and check its content
                            file_content = await read_file(
                                execution_result.repository_path,
                                file_name,
                                max_lines=100
                            )
                            criterion_result["evidence"].append(f"File {file_name} exists.")
                            
                            # If critertion mentions file content or specific elements
                            if "content" in criterion.lower() or "contains" in criterion.lower():
                                # Look for keywords in content
                                content_keywords = ["function", "class", "test", "import", "def"]
                                for keyword in content_keywords:
                                    if keyword in criterion.lower() and keyword in file_content.lower():
                                        criterion_result["evidence"].append(
                                            f"File {file_name} contains {keyword} as required."
                                        )
                            
                            # Success for this file
                            criterion_result["passed"] = True
                        except ValueError:
                            criterion_result["evidence"].append(f"File {file_name} does not exist.")
                            criterion_result["passed"] = False
                            break
                
                # Check for test-related criteria
                elif any(word in criterion.lower() for word in ["test", "unittest", "pytest"]):
                    # Find test files in the repository
                    all_files = []
                    for root, dirs, files in os.walk(execution_result.repository_path):
                        for file in files:
                            if file.startswith("test_") or file.endswith("_test.py"):
                                all_files.append(os.path.join(root, file))
                    
                    if all_files:
                        criterion_result["evidence"].append(f"Found {len(all_files)} test files.")
                        
                        # Try to run the tests
                        try:
                            test_output, _ = await run_terminal_cmd(
                                command=["python", "-m", "unittest", "discover"],
                                cwd=execution_result.repository_path
                            )
                            criterion_result["evidence"].append(f"Tests ran successfully: {test_output}")
                            criterion_result["passed"] = True
                        except Exception as e:
                            criterion_result["evidence"].append(f"Tests failed: {str(e)}")
                            criterion_result["passed"] = False
                    else:
                        criterion_result["evidence"].append("No test files found.")
                        criterion_result["passed"] = False
                
                # Generic code quality criteria
                elif any(word in criterion.lower() for word in ["documenta", "comment", "docstring"]):
                    # Look for Python files with good documentation
                    py_files = []
                    for root, dirs, files in os.walk(execution_result.repository_path):
                        for file in files:
                            if file.endswith(".py"):
                                py_files.append(os.path.join(root, file))
                    
                    if py_files:
                        # Sample some files to check for documentation
                        well_documented_count = 0
                        sample_size = min(5, len(py_files))
                        sampled_files = random.sample(py_files, sample_size)
                        
                        for py_file in sampled_files:
                            try:
                                rel_path = os.path.relpath(py_file, execution_result.repository_path)
                                content = await read_file(
                                    execution_result.repository_path, 
                                    rel_path,
                                    max_lines=200
                                )
                                
                                # Check for docstrings and comments
                                if '"""' in content or "'''" in content:
                                    well_documented_count += 1
                                    criterion_result["evidence"].append(
                                        f"File {rel_path} contains docstrings."
                                    )
                            except Exception:
                                pass
                        
                        # Consider criterion passed if majority of files have documentation
                        if well_documented_count >= sample_size / 2:
                            criterion_result["passed"] = True
                            criterion_result["evidence"].append(
                                f"{well_documented_count}/{sample_size} sampled files are well-documented."
                            )
                        else:
                            criterion_result["passed"] = False
                            criterion_result["evidence"].append(
                                f"Only {well_documented_count}/{sample_size} sampled files are well-documented."
                            )
                    else:
                        criterion_result["evidence"].append("No Python files found to evaluate documentation.")
                        criterion_result["passed"] = False
                
                # Default fallback for other criteria types
                else:
                    # Check for the presence of any changed files
                    changed_files = []
                    try:
                        output, _ = await run_terminal_cmd(
                            command=["git", "diff", "--name-only", "main"],
                            cwd=execution_result.repository_path
                        )
                        changed_files = output.strip().split("\n") if output.strip() else []
                    except Exception:
                        pass
                    
                    if changed_files:
                        criterion_result["evidence"].append(f"Changes made to {len(changed_files)} files.")
                        criterion_result["passed"] = True
                    else:
                        criterion_result["evidence"].append("No changes detected in the repository.")
                        criterion_result["passed"] = False
                
                # Generate reasoning for this criterion
                criterion_result["reasoning"] = self._generate_criterion_reasoning(
                    criterion, 
                    criterion_result["passed"],
                    criterion_result["evidence"]
                )
                
                # Add to overall results
                criteria_results.append(criterion_result)
                if criterion_result["passed"]:
                    total_score += 1.0
            
            # Calculate final score
            score = total_score / len(goal.validation_criteria) if goal.validation_criteria else 0.0
            success = score >= goal.success_threshold
            
            # Generate overall reasoning
            reasoning = self._generate_reasoning(criteria_results, score, goal.success_threshold)
            
            return ValidationResult(
                success=success,
                score=score,
                reasoning=reasoning,
                criteria_results=criteria_results,
                git_hash=execution_result.git_hash,
                branch_name=execution_result.branch_name
            )
            
        finally:
            # Always switch back to main branch
            try:
                await run_terminal_cmd(
                    command=["git", "checkout", "main"],
                    cwd=execution_result.repository_path
                )
            except:
                pass
    
    def _generate_criterion_reasoning(self, criterion: str, passed: bool, evidence: List[str]) -> str:
        """Generate detailed reasoning for a single criterion validation."""
        if passed:
            return f"Criterion satisfied: {criterion}\nEvidence: {'; '.join(evidence)}"
        else:
            return f"Criterion not satisfied: {criterion}\nEvidence: {'; '.join(evidence)}"
    
    def _generate_reasoning(self, criteria_results: List[Dict[str, Any]], 
                          score: float, threshold: float) -> str:
        """Generate a human-readable reasoning for the validation result."""
        passed_count = sum(1 for result in criteria_results if result["passed"])
        total_count = len(criteria_results)
        
        reasoning = []
        reasoning.append(f"Validation {'passed' if score >= threshold else 'failed'} with score {score:.2f}/{threshold:.2f}")
        reasoning.append(f"Satisfied {passed_count}/{total_count} criteria")
        
        # Add details for failed criteria
        if passed_count < total_count:
            reasoning.append("\nFailed criteria:")
            for result in criteria_results:
                if not result["passed"]:
                    reasoning.append(f"- {result['criterion']}")
                    reasoning.append(f"  Reason: {result['reasoning']}")
        
        return "\n".join(reasoning) 