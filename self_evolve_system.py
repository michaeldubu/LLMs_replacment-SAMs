# self_evolve_system.py - Autonomous self-improvement system for SAM

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import time
import logging
import os
import threading
import re
import random
import uuid
import subprocess
import tempfile
import ast
import sympy
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter, deque

logger = logging.getLogger("SAM.SelfEvolve")

@dataclass
class TaskResult:
    """Represents the result of a self-generated task"""
    task_id: str
    task_type: str
    task_prompt: str
    generated_solution: str
    is_correct: bool
    verification_details: Dict[str, Any]
    generated_at: float = field(default_factory=time.time)
    concepts_involved: List[int] = field(default_factory=list)
    reward_signal: float = 0.0
    reasoning_mode: str = "deduction"  # deduction, abduction, induction, trial_error
    complexity_level: int = 1
    steps_taken: List[str] = field(default_factory=list)

class SelfEvolveSystem:
    """Autonomous self-improvement system for SAM"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or model.config
        
        # Task generation parameters
        self.max_task_complexity = 10  # Will increase as system improves
        self.current_complexity = 1
        self.success_rate_threshold = 0.7  # When to increase complexity
        
        # History tracking
        self.task_history = []
        self.success_rates = defaultdict(lambda: deque(maxlen=100))  # Recent success rates by task type
        self.concept_performance = defaultdict(lambda: defaultdict(float))  # concept_id -> task_type -> performance
        
        # Current focus areas (dynamic)
        self.focus_areas = ["code_basic", "math_arithmetic", "logic_basic"]
        
        # Threading control
        self.evolve_thread = None
        self.stop_evolving = threading.Event()
        self.evolving_active = False
        
        # Maps reasoning modes to probability distributions
        self.reasoning_mode_probs = {
            "code_basic": {"deduction": 0.4, "trial_error": 0.3, "abduction": 0.2, "induction": 0.1},
            "code_advanced": {"deduction": 0.3, "abduction": 0.3, "induction": 0.3, "trial_error": 0.1},
            "math_arithmetic": {"deduction": 0.5, "induction": 0.3, "abduction": 0.1, "trial_error": 0.1},
            "math_algebra": {"deduction": 0.4, "abduction": 0.3, "induction": 0.2, "trial_error": 0.1},
            "logic_basic": {"deduction": 0.6, "abduction": 0.2, "induction": 0.1, "trial_error": 0.1},
            "logic_advanced": {"deduction": 0.4, "induction": 0.3, "abduction": 0.2, "trial_error": 0.1}
        }
        
        # Task generators and verifiers
        self.task_generators = {
            "code_basic": self._generate_code_basic_task,
            "code_advanced": self._generate_code_advanced_task,
            "math_arithmetic": self._generate_math_arithmetic_task,
            "math_algebra": self._generate_math_algebra_task,
            "logic_basic": self._generate_logic_basic_task,
            "logic_advanced": self._generate_logic_advanced_task
        }
        
        self.solution_verifiers = {
            "code_basic": self._verify_code_solution,
            "code_advanced": self._verify_code_solution,
            "math_arithmetic": self._verify_math_arithmetic_solution,
            "math_algebra": self._verify_math_algebra_solution,
            "logic_basic": self._verify_logic_solution,
            "logic_advanced": self._verify_logic_solution
        }
        
        # Temp directory for code execution
        self.temp_dir = tempfile.mkdtemp(prefix="sam_evolve_")
        
        # Statistics tracking
        self.stats = {
            "total_tasks_generated": 0,
            "total_tasks_solved": 0,
            "total_concepts_reinforced": 0,
            "complexity_progression": [(time.time(), self.current_complexity)],
            "focus_area_history": [(time.time(), self.focus_areas.copy())],
            "reasoning_mode_usage": defaultdict(int)
        }
    
    def start_evolution(self, interval_minutes=10):
        """Start background evolution process"""
        if self.evolving_active:
            return False
        
        self.stop_evolving.clear()
        self.evolving_active = True
        
        def evolve_loop():
            while not self.stop_evolving.is_set():
                try:
                    # Set model to eval mode temporarily
                    was_training = self.model.training
                    self.model.eval()
                    
                    # Run evolution cycle
                    self.evolve_cycle(duration_minutes=interval_minutes)
                    
                    # Restore model mode
                    if was_training:
                        self.model.train()
                    
                    # Sleep between cycles
                    for _ in range(int(interval_minutes * 60)):
                        if self.stop_evolving.is_set():
                            break
                        time.sleep(1)
                
                except Exception as e:
                    logger.error(f"Error in evolution loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(60)  # Sleep for a minute if there's an error
        
        self.evolve_thread = threading.Thread(target=evolve_loop)
        self.evolve_thread.daemon = True
        self.evolve_thread.start()
        
        logger.info(f"Started self-evolution thread with {interval_minutes} minute interval")
        return True
    
    def stop_evolution(self):
        """Stop background evolution process"""
        if not self.evolving_active:
            return False
        
        self.stop_evolving.set()
        if self.evolve_thread:
            self.evolve_thread.join(timeout=10)
        
        self.evolving_active = False
        logger.info("Stopped self-evolution")
        return True
    
    def evolve_cycle(self, duration_minutes=5, task_count=None):
        """Run a self-evolution cycle for the specified duration or task count"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        task_results = []
        task_num = 0
        
        while (time.time() < end_time and (task_count is None or task_num < task_count)) and not self.stop_evolving.is_set():
            # 1. Select a task type based on current focus
            task_type = self._select_task_type()
            
            # 2. Select reasoning mode for this task
            reasoning_mode = self._select_reasoning_mode(task_type)
            
            # 3. Generate a task
            try:
                task = self._generate_task(task_type, reasoning_mode)
                
                # 4. Solve the task
                solution, steps = self._solve_task(task, reasoning_mode)
                
                # 5. Verify the solution
                is_correct, verification_details = self._verify_solution(task_type, task, solution)
                
                # 6. Calculate reward signal
                reward = self._calculate_reward(task_type, is_correct, verification_details)
                
                # 7. Create result record
                result = TaskResult(
                    task_id=str(uuid.uuid4()),
                    task_type=task_type,
                    task_prompt=task["prompt"],
                    generated_solution=solution,
                    is_correct=is_correct,
                    verification_details=verification_details,
                    concepts_involved=task.get("concepts_involved", []),
                    reward_signal=reward,
                    reasoning_mode=reasoning_mode,
                    complexity_level=task.get("complexity", 1),
                    steps_taken=steps
                )
                
                # 8. Apply learning from result
                self._apply_learning(result)
                
                # Track task
                task_results.append(result)
                task_num += 1
                
                # Update statistics
                self.stats["total_tasks_generated"] += 1
                if is_correct:
                    self.stats["total_tasks_solved"] += 1
                self.stats["reasoning_mode_usage"][reasoning_mode] += 1
                
                # Update success rate tracking
                self.success_rates[task_type].append(1 if is_correct else 0)
                
                # Conditionally adjust complexity
                self._check_complexity_adjustment()
                
                # Update focus areas periodically
                if task_num % 10 == 0:
                    self._update_focus_areas()
            
            except Exception as e:
                logger.error(f"Error in task generation/solution cycle: {e}")
                logger.error(traceback.format_exc())
        
        # Record final stats before returning
        evolution_summary = {
            "duration_minutes": (time.time() - start_time) / 60,
            "tasks_completed": task_num,
            "success_rate": sum(1 for r in task_results if r.is_correct) / max(1, len(task_results)),
            "complexity_level": self.current_complexity,
            "focus_areas": self.focus_areas.copy(),
            "task_type_distribution": Counter([r.task_type for r in task_results]),
            "reasoning_modes_used": Counter([r.reasoning_mode for r in task_results])
        }
        
        logger.info(f"Evolution cycle completed: {task_num} tasks, " 
                   f"{evolution_summary['success_rate']*100:.1f}% success rate")
        
        return evolution_summary
    
    def _select_task_type(self):
        """Select a task type based on current focus areas"""
        return random.choice(self.focus_areas)
    
    def _select_reasoning_mode(self, task_type):
        """Select a reasoning mode for the given task type"""
        probs = self.reasoning_mode_probs.get(task_type, 
                  {"deduction": 0.4, "abduction": 0.2, "induction": 0.2, "trial_error": 0.2})
        
        modes = list(probs.keys())
        weights = list(probs.values())
        
        return random.choices(modes, weights=weights, k=1)[0]
    
    def _generate_task(self, task_type, reasoning_mode):
        """Generate a task of the specified type using the specified reasoning mode"""
        generator = self.task_generators.get(task_type)
        
        if not generator:
            raise ValueError(f"No generator available for task type: {task_type}")
        
        # Generate the task
        task = generator(self.current_complexity, reasoning_mode)
        
        # Add metadata
        task["type"] = task_type
        task["generated_at"] = time.time()
        task["reasoning_mode"] = reasoning_mode
        
        return task
    
    def _solve_task(self, task, reasoning_mode):
        """Solve the given task using the specified reasoning mode"""
        prompt = self._create_solution_prompt(task, reasoning_mode)
        
        # Generate solution with appropriate settings
        temperature = 0.8 if reasoning_mode == "trial_error" else 0.2  # Higher temp for trial/error
        max_length = 1000 if task["type"].startswith("code") else 500
        
        # Use the model to generate a solution
        with torch.no_grad():
            solution = self.model.generate(
                input_text=prompt,
                max_length=max_length,
                temperature=temperature,
                private_context=True  # Mark as private to avoid hive mind sharing
            )
        
        # Extract steps and solution from the generated text
        steps, clean_solution = self._extract_solution_steps(solution, task["type"])
        
        return clean_solution, steps
    
    def _create_solution_prompt(self, task, reasoning_mode):
        """Create a prompt for solving the task based on reasoning mode"""
        prompt_templates = {
            "deduction": (
                "I need to solve this problem using deductive reasoning, working from general principles to a specific solution.\n\n"
                "Task: {prompt}\n\n"
                "I'll solve this step by step, applying known rules and logical deduction to reach the correct solution."
            ),
            "abduction": (
                "I need to solve this problem using abductive reasoning, making my best explanation from incomplete information.\n\n"
                "Task: {prompt}\n\n"
                "I'll work backwards from the expected outcome, making hypotheses about what could explain it, "
                "and find the most plausible solution."
            ),
            "induction": (
                "I need to solve this problem using inductive reasoning, looking for patterns and making generalizations.\n\n"
                "Task: {prompt}\n\n"
                "I'll examine specific examples or instances, identify patterns, and form a general solution."
            ),
            "trial_error": (
                "I need to solve this problem through trial and error, testing different approaches systematically.\n\n"
                "Task: {prompt}\n\n"
                "I'll generate multiple potential solutions, test each one, and refine my approach based on what works and what doesn't."
            )
        }
        
        # Select template based on reasoning mode
        template = prompt_templates.get(reasoning_mode, prompt_templates["deduction"])
        
        # Format the prompt
        return template.format(prompt=task["prompt"])
    
    def _extract_solution_steps(self, solution_text, task_type):
        """Extract solution steps and clean solution from generated text"""
        # Default extraction logic
        steps = []
        clean_solution = solution_text
        
        # Look for step markers (numbered steps or step headers)
        step_matches = re.findall(r'(Step \d+:?|^\d+\.|^- Step)(.+?)(?=Step \d+:|^\d+\.|^- Step|$)', 
                                solution_text, re.MULTILINE | re.DOTALL)
        
        if step_matches:
            steps = [step[1].strip() for step in step_matches]
            
            # For code tasks, extract code blocks
            if task_type.startswith("code"):
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', solution_text, re.DOTALL)
                if code_blocks:
                    clean_solution = code_blocks[-1].strip()  # Take the last code block as the solution
            
        return steps, clean_solution
    
    def _verify_solution(self, task_type, task, solution):
        """Verify if the solution is correct"""
        verifier = self.solution_verifiers.get(task_type)
        
        if not verifier:
            raise ValueError(f"No verifier available for task type: {task_type}")
        
        return verifier(task, solution)
    
    def _calculate_reward(self, task_type, is_correct, verification_details):
        """Calculate reward signal based on task difficulty and correctness"""
        base_reward = 1.0 if is_correct else -0.2
        
        # Adjust based on task type and complexity
        if task_type.startswith("code"):
            # Code rewards are higher for functional code with good style
            if is_correct:
                code_quality = verification_details.get("code_quality", 0.5)
                efficiency = verification_details.get("efficiency", 0.5)
                base_reward = base_reward * (1 + 0.5 * code_quality + 0.5 * efficiency)
        
        elif task_type.startswith("math"):
            # Math rewards are higher for efficient solutions
            if is_correct:
                steps_efficiency = verification_details.get("steps_efficiency", 0.5)
                base_reward = base_reward * (1 + steps_efficiency)
        
        # Scale by complexity
        complexity_factor = task.get("complexity", 1) / 5  # Normalize to [0.2, 2.0]
        reward = base_reward * (1 + complexity_factor)
        
        return reward
    
    def _apply_learning(self, result):
        """Apply learning from task results to improve the model"""
        # 1. Record task in history
        self.task_history.append(result)
        
        # 2. Update concept performance tracking
        for concept_id in result.concepts_involved:
            self.concept_performance[concept_id][result.task_type] += result.reward_signal
        
        # 3. Reinforce concepts based on reward signal
        if result.concepts_involved and result.reward_signal != 0:
            reinforced_count = self._reinforce_concepts(
                result.concepts_involved, 
                result.reward_signal,
                result.task_type
            )
            self.stats["total_concepts_reinforced"] += reinforced_count
        
        # 4. Create new synthesis examples for successful tasks
        if result.is_correct and result.reward_signal > 0.5:
            self._synthesize_from_success(result)
    
    def _reinforce_concepts(self, concept_ids, reward_signal, task_type):
        """Reinforce concepts based on reward signal"""
        reinforced_count = 0
        
        for concept_id in concept_ids:
            if concept_id < 0 or concept_id >= self.model.concept_bank.next_concept_id:
                continue  # Skip invalid concepts
            
            # Apply reinforcement proportional to reward signal
            reinforcement = max(-0.5, min(0.5, reward_signal))  # Clamp to [-0.5, 0.5]
            
            # Skip if reinforcement is negligible
            if abs(reinforcement) < 0.05:
                continue
            
            with torch.no_grad():
                # Get current meaning vector
                meaning_vector = self.model.concept_bank.meaning_vectors[concept_id]
                
                # For positive reinforcement, strengthen this concept's connection to the task domain
                if reinforcement > 0:
                    # Find concepts related to this task type
                    related_concepts = self._find_task_related_concepts(task_type)
                    
                    if related_concepts:
                        # Blend with related concept meanings
                        blend_vectors = []
                        for related_id in related_concepts[:5]:  # Limit to top 5 related concepts
                            if related_id != concept_id:  # Avoid self-blending
                                blend_vectors.append(self.model.concept_bank.meaning_vectors[related_id])
                        
                        if blend_vectors:
                            # Average the blend vectors
                            blend = torch.stack(blend_vectors).mean(dim=0)
                            
                            # Apply reinforcement (move slightly toward the blend)
                            updated = (1 - abs(reinforcement)) * meaning_vector + abs(reinforcement) * blend
                            
                            # Normalize and update
                            self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(updated, dim=0)
                            reinforced_count += 1
                
                # For negative reinforcement, slightly modify the vector
                elif reinforcement < 0:
                    # Add a small amount of noise to the vector (exploration)
                    noise = torch.randn_like(meaning_vector) * abs(reinforcement) * 0.1
                    updated = meaning_vector + noise
                    
                    # Normalize and update
                    self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(updated, dim=0)
                    reinforced_count += 1
                
                # Update usage statistics
                self.model.concept_bank.update_concept_usage(concept_id, context=f"self_evolution_{task_type}")
        
        return reinforced_count
    
    def _find_task_related_concepts(self, task_type):
        """Find concepts related to a specific task type"""
        # Get patterns related to this task type
        task_patterns = []
        
        if task_type.startswith("code"):
            patterns = ["function", "def ", "class ", "return", "for ", "while ", "if ", "else ", 
                       "python", "algorithm", "import", "=", "+=", "-=", "==", "!="]
            task_patterns.extend(patterns)
        
        elif task_type.startswith("math"):
            patterns = ["equals", "sum", "product", "divide", "multiply", "subtract", "add", 
                       "equation", "solve", "=", "+", "-", "*", "/", "^", "sqrt", "log"]
            task_patterns.extend(patterns)
        
        elif task_type.startswith("logic"):
            patterns = ["logic", "implies", "and", "or", "not", "if", "then", "therefore", 
                       "conclusion", "premise", "deduction", "valid", "invalid"]
            task_patterns.extend(patterns)
        
        # Find concept IDs for these patterns
        related_concepts = []
        for pattern in task_patterns:
            concept_id = self.model.concept_bank.find_concept_by_source(pattern)
            if concept_id is not None:
                related_concepts.append(concept_id)
        
        return related_concepts
    
    def _synthesize_from_success(self, result):
        """Create synthetic learning examples from successful task solutions"""
        # Only synthesize for successful tasks
        if not result.is_correct:
            return
        
        # Create a new dream synthesis entry in the model's dreaming component
        if hasattr(self.model, "dreaming") and hasattr(self.model.dreaming, "synthesis_history"):
            task_text = f"Task: {result.task_prompt}\n\nSolution: {result.generated_solution}"
            
            self.model.dreaming.synthesis_history.append({
                "type": f"self_evolution_{result.task_type}",
                "seed": result.task_prompt[:50] + "...",
                "generated": task_text,
                "timestamp": time.time()
            })
    
    def _check_complexity_adjustment(self):
        """Check if complexity should be adjusted based on success rates"""
        # Only check after we have enough history
        if not any(len(rates) >= 20 for rates in self.success_rates.values()):
            return
        
        # Calculate overall success rate
        total_successes = sum(sum(rates) for rates in self.success_rates.values())
        total_tasks = sum(len(rates) for rates in self.success_rates.values())
        overall_success_rate = total_successes / total_tasks if total_tasks > 0 else 0
        
        # Adjust complexity based on success rate
        if overall_success_rate > self.success_rate_threshold and self.current_complexity < self.max_task_complexity:
            self.current_complexity += 1
            self.stats["complexity_progression"].append((time.time(), self.current_complexity))
            logger.info(f"Increasing complexity to level {self.current_complexity} "
                       f"(success rate: {overall_success_rate:.2f})")
        elif overall_success_rate < 0.3 and self.current_complexity > 1:
            self.current_complexity -= 1
            self.stats["complexity_progression"].append((time.time(), self.current_complexity))
            logger.info(f"Decreasing complexity to level {self.current_complexity} "
                       f"(success rate: {overall_success_rate:.2f})")
    
    def _update_focus_areas(self):
        """Update focus areas based on performance"""
        # Calculate success rates for each task type
        success_by_type = {}
        for task_type, rates in self.success_rates.items():
            if rates:
                success_by_type[task_type] = sum(rates) / len(rates)
        
        if not success_by_type:
            return  # Not enough data to update focus
        
        # Find underperforming areas
        underperforming = [task_type for task_type, rate in success_by_type.items() 
                         if rate < self.success_rate_threshold and task_type in self.focus_areas]
        
        # Find high-performing areas
        high_performing = [task_type for task_type, rate in success_by_type.items()
                         if rate > 0.9 and task_type in self.focus_areas]
        
        # Adjust focus areas
        if underperforming:
            # Increase focus on underperforming areas
            self.focus_areas = list(set(self.focus_areas) - set(high_performing))
            self.focus_areas.extend(underperforming)
        elif len(self.focus_areas) < 3 and high_performing:
            # If we've mastered current focus, add higher complexity versions
            for task_type in high_performing:
                if task_type.endswith("_basic") and f"{task_type[:-6]}_advanced" not in self.focus_areas:
                    self.focus_areas.append(f"{task_type[:-6]}_advanced")
            
            # Ensure we don't have too many focus areas
            if len(self.focus_areas) > 4:
                # Keep underperforming areas and a mix of basic/advanced
                self.focus_areas = underperforming + random.sample(
                    [t for t in self.focus_areas if t not in underperforming], 
                    min(3, len(self.focus_areas) - len(underperforming))
                )
        
        # Track focus area changes
        self.stats["focus_area_history"].append((time.time(), self.focus_areas.copy()))
    
    #
    # Task generators
    #
    
    def _generate_code_basic_task(self, complexity, reasoning_mode):
        """Generate a basic coding task"""
        # Define task templates based on complexity
        task_templates = {
            1: [
                {"prompt": "Write a Python function called `hello_world` that prints 'Hello, World!'"},
                {"prompt": "Write a Python function called `add_numbers` that takes two parameters and returns their sum."},
                {"prompt": "Write a Python function called `is_even` that checks if a number is even."},
            ],
            2: [
                {"prompt": "Write a Python function called `reverse_string` that reverses a string."},
                {"prompt": "Write a Python function called `find_max` that returns the largest number in a list."},
                {"prompt": "Write a Python function called `count_vowels` that counts the vowels in a string."},
            ],
            3: [
                {"prompt": "Write a Python function called `is_palindrome` that checks if a string is a palindrome."},
                {"prompt": "Write a Python function called `fibonacci` that returns the nth Fibonacci number."},
                {"prompt": "Write a Python function called `factorial` that calculates the factorial of a number."},
            ],
            4: [
                {"prompt": "Write a Python function called `find_primes` that returns all prime numbers up to n."},
                {"prompt": "Write a Python function called `binary_search` that implements binary search on a sorted list."},
                {"prompt": "Write a Python function called `merge_sorted_lists` that merges two sorted lists into one sorted list."},
            ],
            5: [
                {"prompt": "Write a Python function called `valid_parentheses` that checks if a string of parentheses is valid."},
                {"prompt": "Write a Python function called `remove_duplicates` that removes duplicates from a list while preserving order."},
                {"prompt": "Write a Python function called `group_anagrams` that groups anagrams from a list of strings."},
            ]
        }
        
        # For higher complexity levels, use complexity 5 tasks with added constraints
        if complexity > 5:
            base_complexity = 5
            # Add additional constraints or challenges
            templates = task_templates[base_complexity]
            selected = random.choice(templates)
            
            # Add extra constraints based on complexity
            constraints = [
                "without using any built-in functions except basic operators",
                "in O(n) time complexity and O(1) space complexity",
                "using a recursive approach",
                "using only list comprehensions, no explicit loops",
                "with comprehensive error handling for all edge cases"
            ]
            
            # Select random constraints based on complexity level
            num_constraints = min(complexity - base_complexity, len(constraints))
            selected_constraints = random.sample(constraints, num_constraints)
            
            # Create new prompt with constraints
            prompt = selected["prompt"] + " " + " and ".join(selected_constraints)
            task = {"prompt": prompt, "complexity": complexity}
        else:
            # Use template for the exact complexity level
            templates = task_templates.get(complexity, task_templates[1])
            task = random.choice(templates)
            task["complexity"] = complexity
        
        # Create expected tests for verification
        function_name = re.search(r'called `(\w+)`', task["prompt"])
        if function_name:
            function_name = function_name.group(1)
            task["verification"] = self._generate_code_tests(function_name, task["prompt"], complexity)
        
        # Find concepts related to this task
        task["concepts_involved"] = self._extract_task_concepts(task["prompt"])
        
        return task
    
    def _generate_code_advanced_task(self, complexity, reasoning_mode):
        """Generate an advanced coding task"""
        # Adjust complexity - advanced tasks start at complexity 5
        task_complexity = max(5, complexity)
        
        # Define advanced task templates
        advanced_templates = [
            {"prompt": f"Implement a {random.choice(['Stack', 'Queue', 'LinkedList', 'BinaryTree'])} class in Python with standard methods."},
            {"prompt": "Write a function to detect a cycle in a linked list."},
            {"prompt": "Implement a function that performs in-order traversal of a binary tree."},
            {"prompt": "Create a function that implements the quick sort algorithm."},
            {"prompt": "Write a function to find the longest common subsequence of two strings."},
            {"prompt": "Implement a class for a simple LRU cache with get and put methods."},
            {"prompt": "Create a trie data structure for efficient string storage and retrieval."},
            {"prompt": "Implement a function to solve the knapsack problem using dynamic programming."}
        ]
        
        # Select a template
        task = random.choice(advanced_templates)
        task["complexity"] = task_complexity
        
        # Create verification criteria
        task["verification"] = {
            "test_cases": [
                # Will be filled by the verification function
            ],
            "complexity_requirements": {
                "time": "O(n)" if task_complexity < 8 else "O(log n)",
                "space": "O(n)" if task_complexity < 10 else "O(1)"
            },
            "style_requirements": [
                "clear variable names",
                "proper docstrings",
                "efficient implementation"
            ]
        }
        
        # Extract concepts
        task["concepts_involved"] = self._extract_task_concepts(task["prompt"])
        
        return task
    
    def _generate_math_arithmetic_task(self, complexity, reasoning_mode):
        """Generate a basic arithmetic task"""
        operations = ["+", "-", "*", "/"]
        
        # Adjust number ranges based on complexity
        num_range = {
            1: (1, 10),
            2: (10, 100),
            3: (100, 1000),
            4: (1000, 10000),
            5: (10000, 100000)
        }
        
        # Get range for this complexity
        min_val, max_val = num_range.get(min(complexity, 5), (1, 10))
        
        # Generate numbers
        num_operations = min(complexity, 5)
        numbers = [random.randint(min_val, max_val) for _ in range(num_operations + 1)]
        
        # Generate operations
        ops = [random.choice(operations) for _ in range(num_operations)]
        
        # Create expression
        expression = str(numbers[0])
        for i in range(num_operations):
            expression += f" {ops[i]} {numbers[i+1]}"
        
        # For division, ensure clean division if complexity <= 2
        if complexity <= 2 and "/" in ops:
            # Recalculate the expression to ensure clean division
            parts = expression.split()
            for i in range(1, len(parts), 2):
                if parts[i] == "/":
                    # Ensure divisor isn't 0
                    if int(parts[i+1]) == 0:
                        parts[i+1] = "1"
                    
                    # Ensure clean division for lower complexity
                    dividend = int(parts[i-1])
                    divisor = int(parts[i+1])
                    parts[i-1] = str(dividend * divisor)
            
            expression = " ".join(parts)
        
        # Handle different reasoning modes
        if reasoning_mode == "abduction":
            # For abduction, provide the answer and ask for part of the expression
            answer = eval(expression)
            parts = expression.split()
            blank_idx = random.randint(0, len(parts) - 1)
            blank_value = parts[blank_idx]
            parts[blank_idx] = "___"
            expression_with_blank = " ".join(parts)
            
            prompt = f"Fill in the blank to make this equation correct: {expression_with_blank} = {answer}"
            answer = blank_value
        else:
            # For other reasoning modes, ask for the answer
            prompt = f"Calculate the result of this expression: {expression}"
            answer = eval(expression)
        
        # Create task
        task = {
            "prompt": prompt,
            "verification": {
                "answer": answer,
                "expression": expression
            },
            "complexity": complexity
        }
        
        # Extract concepts
        task["concepts_involved"] = self._extract_task_concepts(prompt)
        
        return task
    
    def _generate_math_algebra_task(self, complexity, reasoning_mode):
        """Generate an algebra task"""
        # Create task based on complexity
        if complexity <= 3:
            # Simple linear equations
            a = random.randint(1, 5) if complexity == 1 else random.randint(1, 20)
            b = random.randint(1, 10) if complexity <= 2 else random.randint(1, 50)
            c = a * random.randint(1, 5) if complexity <= 2 else random.randint(1, 100)
            
            # Create equation: ax + b = c
            equation = f"{a}x + {b} = {c}"
            answer = (c - b) / a
            
            prompt = f"Solve for x: {equation}"
        elif complexity <= 6:
            # Quadratic equations
            a = random.randint(1, 3) if complexity <= 4 else random.randint(1, 5)
            
            # For simplicity, create factors (x-p)(x-q) with integer roots
            p = random.randint(-10, 10)
            q = random.randint(-10, 10)
            
            # Create equation: a(x-p)(x-q) = 0 => ax² - a(p+q)x + apq = 0
            b = -a * (p + q)
            c = a * p * q
            
            equation = f"{a}x² + {b}x + {c} = 0"
            answers = [p, q]
            
            prompt = f"Find all solutions for x: {equation}"
        else:
            # Systems of equations with two variables
            a1 = random.randint(1, 10)
            b1 = random.randint(1, 10)
            c1 = a1 * random.randint(1, 5) + b1 * random.randint(1, 5)
            
            a2 = random.randint(1, 10)
            b2 = random.randint(1, 10)
            c2 = a2 * random.randint(1, 5) + b2 * random.randint(1, 5)
            
            # Create system: a1x + b1y = c1, a2x + b2y = c2
            equation1 = f"{a1}x + {b1}y = {c1}"
            equation2 = f"{a2}x + {b2}y = {c2}"
            
            # Calculate answers
            determinant = a1 * b2 - a2 * b1
            if determinant != 0:
                x = (c1 * b2 - c2 * b1) / determinant
                y = (a1 * c2 - a2 * c1) / determinant
                answers = {"x": x, "y": y}
            else:
                # Handle degenerate case
                answers = "No unique solution"
            
            prompt = f"Solve the system of equations:\n{equation1}\n{equation2}"
        
        # Create task
        task = {
            "prompt": prompt,
            "verification": {
                "answer": answers,
                "equation": equation if complexity <= 6 else [equation1, equation2]
            },
            "complexity": complexity
        }
        
        # Extract concepts
        task["concepts_involved"] = self._extract_task_concepts(prompt)
        
        return task
    
    def _generate_logic_basic_task(self, complexity, reasoning_mode):
        """Generate a basic logic task"""
        if complexity <= 2:
            # Simple boolean logic
            operators = ["AND", "OR", "NOT"]
            values = [True, False]
            
            if complexity == 1:
                # Single operation
                op = random.choice(operators[:2])  # AND or OR for simplicity
                a, b = random.choice(values), random.choice(values)
                
                expression = f"{a} {op} {b}"
                answer = eval(f"{a} {op.lower()} {b}")
                
                prompt = f"Evaluate the boolean expression: {expression}"
            else:
                # Two operations
                op1, op2 = random.sample(operators, 2)
                a, b, c = [random.choice(values) for _ in range(3)]
                
                if op1 == "NOT":
                    expression = f"({op1} {a}) {op2} {b}"
                    answer = eval(f"(not {a}) {op2.lower()} {b}")
                elif op2 == "NOT":
                    expression = f"{a} {op1} ({op2} {b})"
                    answer = eval(f"{a} {op1.lower()} (not {b})")
                else:
                    expression = f"{a} {op1} {b} {op2} {c}"
                    answer = eval(f"{a} {op1.lower()} {b} {op2.lower()} {c}")
                
                prompt = f"Evaluate the boolean expression: {expression}"
        elif complexity <= 4:
            # Logical deduction with simple rules
            actors = ["Alice", "Bob", "Charlie", "Diana", "Evan"]
            objects = ["apple", "banana", "carrot", "donut", "egg"]
            locations = ["kitchen", "living room", "bedroom", "bathroom", "garden"]
            
            # Select a subset
            num_items = min(complexity, 3)
            selected_actors = random.sample(actors, num_items)
            selected_objects = random.sample(objects, num_items)
            
            # Create rules
            rules = []
            facts = {}
            
            for i in range(num_items):
                actor = selected_actors[i]
                obj = selected_objects[i]
                facts[actor] = obj
                
                # Create rule
                if i < num_items - 1:
                    rules.append(f"{actor} has the {obj}.")
            
            # Add question
            missing_actor = selected_actors[-1]
            missing_object = selected_objects[-1]
            
            prompt = f"Based on these facts:\n" + "\n".join(rules)
            prompt += f"\n\nIf everyone has exactly one item, and all items are different, what does {missing_actor} have?"
            
            answer = missing_object
        else:
            # Logic grid puzzle
            categories = [
                ["Alice", "Bob", "Charlie", "Diana"],
                ["red", "blue", "green", "yellow"],
                ["dog", "cat", "bird", "fish"]
            ]
            
            # Create a valid solution
            solution = []
            for i in range(len(categories[0])):
                item = [c[i] for c in categories]
                solution.append(item)
            
            # Shuffle solution
            random.shuffle(solution)
            
            # Create clues
            clues = []
            
            # Direct clues
            clues.append(f"{solution[0][0]} has the {solution[0][2]}.")
            
            # Negative clues
            person = random.choice(categories[0])
            wrong_color = random.choice([c for c in categories[1] if [person, c] not in [[s[0], s[1]] for s in solution]])
            clues.append(f"{person} does not have the {wrong_color} item.")
            
            # Relative clues
            for i in range(min(complexity - 5, 2)):
                s1, s2 = random.sample(solution, 2)
                clues.append(f"The person with the {s1[2]} also has the {s1[1]} item.")
                clues.append(f"The {s2[1]} item is not owned by the person with the {s2[2]}.")
            
            # Add question
            target = random.choice(solution)
            
            prompt = f"Logic puzzle: Four people each have one pet and one color associated with them. Based on these clues:\n"
            prompt += "\n".join(clues)
            prompt += f"\n\nWhat color is associated with the person who has the {target[2]}?"
            
            answer = target[1]
        
        # Create task
        task = {
            "prompt": prompt,
            "verification": {
                "answer": answer
            },
            "complexity": complexity
        }
        
        # Extract concepts
        task["concepts_involved"] = self._extract_task_concepts(prompt)
        
        return task
    
    def _generate_logic_advanced_task(self, complexity, reasoning_mode):
        """Generate an advanced logic task"""
        # Advanced logic tasks start at complexity 5
        task_complexity = max(5, complexity)
        
        if task_complexity <= 7:
            # Syllogistic reasoning
            categories = [
                ["All heroes", "Some heroes", "No heroes"],
                ["are brave", "are not brave", "are immortal", "are not immortal"],
                ["All brave individuals", "Some brave individuals", "No brave individuals"],
                ["are heroes", "are not heroes", "are immortal", "are not immortal"]
            ]
            
            # Create premises
            premise1 = random.choice(categories[0]) + " " + random.choice(categories[1]) + "."
            premise2 = random.choice(categories[2]) + " " + random.choice(categories[3]) + "."
            
            # Define valid inferences based on selected premises
            if premise1.startswith("All") and premise2.startswith("All") and premise1.split()[3] == premise2.split()[1]:
                # Valid transitive inference
                conclusion = f"Therefore, all {premise1.split()[1]} are {premise2.split()[3]}."
                is_valid = True
            else:
                # Random conclusion that may or may not be valid
                conclusion = f"Therefore, {random.choice(['all', 'some', 'no'])} {premise1.split()[1]} are {random.choice(premise2.split()[3:])}."
                # Actual validity would require proper syllogistic logic evaluation
                is_valid = random.choice([True, False])
            
            prompt = f"Determine if this syllogism is valid:\n{premise1}\n{premise2}\n{conclusion}"
            
            answer = "Valid" if is_valid else "Invalid"
            
            # Add explanation for verification
            verification = {
                "answer": answer,
                "premises": [premise1, premise2],
                "conclusion": conclusion,
                "reasoning": "Requires syllogistic logic evaluation"
            }
        else:
            # Propositional logic
            variables = ["p", "q", "r"]
            operators = ["∧", "∨", "→", "↔", "¬"]
            
            # Create a formula
            num_operations = min(task_complexity - 5, 4)
            
            # Start with a simple expression
            expr = random.choice(variables)
            
            for _ in range(num_operations):
                op = random.choice(operators)
                
                if op == "¬":
                    # Negation
                    expr = f"¬({expr})"
                else:
                    # Binary operation
                    var = random.choice(variables)
                    expr = f"({expr} {op} {var})"
            
            # Create truth table task
            prompt = f"Complete the truth table for the logical expression: {expr}"
            
            # For verification, we'd need to evaluate the expression
            # In practice, this would require a proper propositional logic evaluator
            verification = {
                "expression": expr,
                "variables": list(set(v for v in expr if v in variables)),
                "correct_evaluation": "Will be evaluated during verification"
            }
            
            answer = "Truth table results"
        
        # Create task
        task = {
            "prompt": prompt,
            "verification": verification,
            "complexity": task_complexity
        }
        
        # Extract concepts
        task["concepts_involved"] = self._extract_task_concepts(prompt)
        
        return task
    
    #
    # Solution verifiers
    #
    
    def _verify_code_solution(self, task, solution):
        """Verify a code solution"""
        try:
            # Clean the solution - extract code if wrapped in markdown
            code_match = re.search(r'```(?:python)?\s*(.*?)```', solution, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                code = solution.strip()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir=self.temp_dir, mode='w') as temp_file:
                temp_filename = temp_file.name
                temp_file.write(code)
            
            # Prepare test script
            test_file = tempfile.NamedTemporaryFile(suffix='_test.py', delete=False, dir=self.temp_dir, mode='w')
            test_filename = test_file.name
            
            # Extract function or class name
            function_name_match = re.search(r'def\s+(\w+)', code) or re.search(r'class\s+(\w+)', code)
            if function_name_match:
                function_name = function_name_match.group(1)
            else:
                # If no function name found in code, try task prompt
                function_name_match = re.search(r'called `(\w+)`', task["prompt"])
                if function_name_match:
                    function_name = function_name_match.group(1)
                else:
                    function_name = "solution_function"  # Default name
            
            # Write test script
            test_file.write(f"import sys\nimport traceback\n\ntry:\n")
            test_file.write(f"    from {os.path.basename(temp_filename)[:-3]} import {function_name}\n\n")
            test_file.write(f"    # Run tests\n")
            
            # Generate tests based on task
            if "verification" in task and "test_cases" in task["verification"]:
                # Use provided test cases
                test_cases = task["verification"]["test_cases"]
            else:
                # Generate basic test cases
                test_cases = self._generate_code_tests(function_name, task["prompt"], task["complexity"])
            
            # Write test cases
            for i, test in enumerate(test_cases):
                test_file.write(f"    # Test {i+1}\n")
                if "input" in test and "expected" in test:
                    inputs_str = ", ".join([repr(inp) for inp in test["input"]]) if isinstance(test["input"], list) else repr(test["input"])
                    test_file.write(f"    result_{i} = {function_name}({inputs_str})\n")
                    test_file.write(f"    assert result_{i} == {repr(test['expected'])}, ")
                    test_file.write(f"f\"Test {i+1} failed: Expected {repr(test['expected'])}, got {{result_{i}}}\"\n")
                elif "code" in test:
                    # Custom test code
                    test_file.write(f"    {test['code']}\n")
            
            test_file.write(f"    print('All tests passed!')\n")
            test_file.write(f"    sys.exit(0)\n")
            test_file.write(f"except Exception as e:\n")
            test_file.write(f"    print(f\"Error: {{str(e)}}\")\n")
            test_file.write(f"    traceback.print_exc()\n")
            test_file.write(f"    sys.exit(1)\n")
            test_file.close()
            
            # Run the test script
            process = subprocess.run(
                [sys.executable, test_filename],
                capture_output=True,
                text=True
            )
            
            # Check results
            is_correct = process.returncode == 0
            
            # Parse output
            output = process.stdout.strip()
            error = process.stderr.strip()
            
            # Assess code quality (basic metrics)
            code_quality = self._assess_code_quality(code)
            
            verification_details = {
                "output": output,
                "error": error,
                "code_quality": code_quality,
                "efficiency": 0.7,  # Default, would need more sophisticated analysis
                "test_cases": test_cases
            }
            
            # Clean up temporary files
            try:
                os.unlink(temp_filename)
                os.unlink(test_filename)
            except:
                pass
            
            return is_correct, verification_details
        
        except Exception as e:
            logger.error(f"Error verifying code solution: {str(e)}")
            logger.error(traceback.format_exc())
            
            return False, {
                "error": str(e),
                "output": "Verification failed",
                "code_quality": 0.0,
                "efficiency": 0.0
            }
    
    def _verify_math_arithmetic_solution(self, task, solution):
        """Verify a math arithmetic solution"""
        try:
            # Extract answer from solution
            # Look for patterns like "answer is X" or "result: X" or just the number
            answer_match = re.search(r'(?:answer|result|solution)\s*(?:is|=|:)\s*(-?\d+\.?\d*)', solution, flags=re.IGNORECASE)
            
            if answer_match:
                extracted_answer = float(answer_match.group(1))
            else:
                # Try to find just a number
                number_match = re.search(r'(?<!\w)(-?\d+\.?\d*)(?!\w)', solution)
                if number_match:
                    extracted_answer = float(number_match.group(1))
                else:
                    # No clear answer found
                    return False, {"error": "No clear answer extracted from solution"}
            
            # Get expected answer
            expected_answer = float(task["verification"]["answer"])
            
            # Check if correct (allow small floating point error)
            is_correct = abs(extracted_answer - expected_answer) < 1e-6
            
            # Analyze solution approach
            steps_found = len(re.findall(r'step|=|\+|-|\*|\/|\d+', solution, re.IGNORECASE))
            steps_efficiency = min(1.0, steps_found / 10)  # Normalize between 0 and 1
            
            return is_correct, {
                "extracted_answer": extracted_answer,
                "expected_answer": expected_answer,
                "steps_efficiency": steps_efficiency
            }
        
        except Exception as e:
            logger.error(f"Error verifying math solution: {str(e)}")
            return False, {"error": str(e)}
    
    def _verify_math_algebra_solution(self, task, solution):
        """Verify a math algebra solution"""
        try:
            # Get expected answer
            expected_answer = task["verification"]["answer"]
            
            # Different verification based on answer type
            if isinstance(expected_answer, list):
                # Multiple solutions (like quadratic)
                # Look for x = value patterns
                solution_matches = re.findall(r'x\s*=\s*(-?\d+\.?\d*)', solution)
                
                if not solution_matches:
                    return False, {"error": "No solutions found in the answer"}
                
                extracted_answers = [float(x) for x in solution_matches]
                
                # Check if all expected solutions are found
                expected_set = set(expected_answer)
                extracted_set = set(extracted_answers)
                
                is_correct = expected_set == extracted_set
                
                verification_details = {
                    "extracted_answers": extracted_answers,
                    "expected_answers": expected_answer,
                    "missing_answers": list(expected_set - extracted_set),
                    "extra_answers": list(extracted_set - expected_set)
                }
            elif isinstance(expected_answer, dict):
                # Multiple variables (like systems of equations)
                # Look for x = value, y = value patterns
                x_match = re.search(r'x\s*=\s*(-?\d+\.?\d*)', solution)
                y_match = re.search(r'y\s*=\s*(-?\d+\.?\d*)', solution)
                
                if not x_match or not y_match:
                    return False, {"error": "Missing solutions for variables"}
                
                extracted_x = float(x_match.group(1))
                extracted_y = float(y_match.group(1))
                
                # Check if correct (allow small floating point error)
                x_correct = abs(extracted_x - expected_answer["x"]) < 1e-6
                y_correct = abs(extracted_y - expected_answer["y"]) < 1e-6
                
                is_correct = x_correct and y_correct
                
                verification_details = {
                    "extracted_x": extracted_x,
                    "expected_x": expected_answer["x"],
                    "extracted_y": extracted_y,
                    "expected_y": expected_answer["y"]
                }
            else:
                # Single solution
                # Look for x = value pattern
                answer_match = re.search(r'x\s*=\s*(-?\d+\.?\d*)', solution)
                
                if not answer_match:
                    return False, {"error": "No solution found in the answer"}
                
                extracted_answer = float(answer_match.group(1))
                
                # Check if correct (allow small floating point error)
                is_correct = abs(extracted_answer - expected_answer) < 1e-6
                
                verification_details = {
                    "extracted_answer": extracted_answer,
                    "expected_answer": expected_answer
                }
            
            # Analyze solution approach
            has_equation_manipulation = re.search(r'(?:(?:add|subtract|multiply|divide).+(?:both|each).+(?:sides|equations))|(?:(?:isolate|solve for).+x)|(?:(?:substitute|eliminate))', solution, re.IGNORECASE) is not None
            has_step_by_step = len(re.findall(r'(?:step \d|first|then|next|finally)', solution, re.IGNORECASE)) > 1
            
            verification_details["steps_efficiency"] = 0.7 if has_equation_manipulation else 0.3
            verification_details["steps_efficiency"] += 0.3 if has_step_by_step else 0.0
            verification_details["steps_efficiency"] = min(1.0, verification_details["steps_efficiency"])
            
            return is_correct, verification_details
        
        except Exception as e:
            logger.error(f"Error verifying algebra solution: {str(e)}")
            return False, {"error": str(e)}
    
    def _verify_logic_solution(self, task, solution):
        """Verify a logic solution"""
        try:
            # Extract expected answer
            expected_answer = task["verification"]["answer"]
            
            # Different verification based on expected answer
            if expected_answer in ["Valid", "Invalid"]:
                # Syllogism validity
                # Look for "valid" or "invalid" in the solution
                valid_match = re.search(r'\b(valid|invalid)\b', solution, re.IGNORECASE)
                
                if not valid_match:
                    return False, {"error": "No clear valid/invalid assessment found"}
                
                extracted_answer = valid_match.group(1).capitalize()
                is_correct = extracted_answer == expected_answer
                
                verification_details = {
                    "extracted_answer": extracted_answer,
                    "expected_answer": expected_answer
                }
            elif isinstance(expected_answer, str) and expected_answer in ["Truth table results"]:
                # Truth table - complex verification
                # In a real implementation, this would parse the truth table and verify it
                is_correct = "true" in solution.lower() and "false" in solution.lower()
                
                verification_details = {
                    "has_truth_table": "table" in solution.lower(),
                    "has_truth_values": "true" in solution.lower() and "false" in solution.lower()
                }
            else:
                # Regular answers (objects, colors, etc.)
                # Look for sentences containing the expected answer
                expected_pattern = re.escape(expected_answer).lower()
                answer_found = re.search(expected_pattern, solution.lower()) is not None
                
                if not answer_found:
                    # Look for different formats (e.g., "the answer is X")
                    answer_match = re.search(r'(?:answer|result|solution)\s*(?:is|=|:)\s*([^\.]+)', solution, flags=re.IGNORECASE)
                    if answer_match:
                        extracted_answer = answer_match.group(1).strip().lower()
                        answer_found = expected_answer.lower() in extracted_answer
                
                is_correct = answer_found
                
                verification_details = {
                    "expected_answer": expected_answer,
                    "answer_found": answer_found
                }
            
            # Analyze solution approach
            has_deduction = re.search(r'(?:therefore|thus|so|implies|because|since)', solution, re.IGNORECASE) is not None
            has_structured_reasoning = len(re.findall(r'(?:step \d|first|then|next|finally)', solution, re.IGNORECASE)) > 1
            
            verification_details["reasoning_quality"] = 0.7 if has_deduction else 0.3
            verification_details["reasoning_quality"] += 0.3 if has_structured_reasoning else 0.0
            verification_details["reasoning_quality"] = min(1.0, verification_details["reasoning_quality"])
            
            return is_correct, verification_details
        
        except Exception as e:
            logger.error(f"Error verifying logic solution: {str(e)}")
            return False, {"error": str(e)}
    
    #
    # Helper methods
    #
    
    def _generate_code_tests(self, function_name, prompt, complexity):
        """Generate test cases for a coding task"""
        # Extract function type from prompt
        is_string = re.search(r'string|text|word', prompt, re.IGNORECASE) is not None
        is_math = re.search(r'add|sum|multiply|subtract|divide|math|calculate|compute', prompt, re.IGNORECASE) is not None
        is_boolean = re.search(r'check|is_|valid|palindrome|prime', prompt, re.IGNORECASE) is not None
        is_list = re.search(r'list|array|sequence|sort|find', prompt, re.IGNORECASE) is not None
        
        # Generate appropriate test cases
        test_cases = []
        
        if "hello_world" in function_name.lower():
            # Special case for hello world
            test_cases.append({"code": f"import io, sys; stdout = sys.stdout; sys.stdout = io.StringIO(); {function_name}(); output = sys.stdout.getvalue().strip(); sys.stdout = stdout; assert output == 'Hello, World!', f\"Expected 'Hello, World!', got '{{output}}'\"" })
        elif is_string:
            # String tests
            if "reverse" in prompt.lower():
                test_cases.append({"input": "hello", "expected": "olleh"})
                test_cases.append({"input": "python", "expected": "nohtyp"})
                test_cases.append({"input": "", "expected": ""})
            elif "palindrome" in prompt.lower():
                test_cases.append({"input": "radar", "expected": True})
                test_cases.append({"input": "hello", "expected": False})
                test_cases.append({"input": "A man a plan a canal Panama", "expected": True})
            elif "vowel" in prompt.lower():
                test_cases.append({"input": "hello", "expected": 2})
                test_cases.append({"input": "sky", "expected": 0})
                test_cases.append({"input": "AEIOU", "expected": 5})
            else:
                # Generic string tests
                test_cases.append({"input": "hello", "expected": None})
                test_cases.append({"input": "", "expected": None})
        elif is_math:
            # Math tests
            if "add" in prompt.lower() or "sum" in prompt.lower():
                test_cases.append({"input": [5, 3], "expected": 8})
                test_cases.append({"input": [-1, 1], "expected": 0})
                test_cases.append({"input": [0, 0], "expected": 0})
            elif "fibonacci" in prompt.lower():
                test_cases.append({"input": 0, "expected": 0})
                test_cases.append({"input": 1, "expected": 1})
                test_cases.append({"input": 6, "expected": 8})
            elif "factorial" in prompt.lower():
                test_cases.append({"input": 0, "expected": 1})
                test_cases.append({"input": 1, "expected": 1})
                test_cases.append({"input": 5, "expected": 120})
            else:
                # Generic math tests
                test_cases.append({"input": 5, "expected": None})
                test_cases.append({"input": 0, "expected": None})
        elif is_boolean:
            # Boolean tests
            if "even" in prompt.lower():
                test_cases.append({"input": 2, "expected": True})
                test_cases.append({"input": 3, "expected": False})
                test_cases.append({"input": 0, "expected": True})
            elif "prime" in prompt.lower():
                test_cases.append({"input": 7, "expected": True})
                test_cases.append({"input": 4, "expected": False})
                test_cases.append({"input": 1, "expected": False})
            elif "valid" in prompt.lower() and "parentheses" in prompt.lower():
                test_cases.append({"input": "()", "expected": True})
                test_cases.append({"input": "(()", "expected": False})
                test_cases.append({"input": "(())", "expected": True})
            else:
                # Generic boolean tests
                test_cases.append({"input": True, "expected": None})
                test_cases.append({"input": False, "expected": None})
        elif is_list:
            # List tests
            if "max" in prompt.lower() or "largest" in prompt.lower():
                test_cases.append({"input": [1, 5, 3], "expected": 5})
                test_cases.append({"input": [-1, -5, -3], "expected": -1})
                test_cases.append({"input": [0], "expected": 0})
            elif "sort" in prompt.lower():
                test_cases.append({"input": [3, 1, 2], "expected": [1, 2, 3]})
                test_cases.append({"input": [], "expected": []})
                test_cases.append({"input": [1], "expected": [1]})
            elif "search" in prompt.lower():
                test_cases.append({"input": [[1, 2, 3, 4, 5], 3], "expected": 2})
                test_cases.append({"input": [[1, 2, 3, 4, 5], 6], "expected": -1})
                test_cases.append({"input": [[], 1], "expected": -1})
            else:
                # Generic list tests
                test_cases.append({"input": [1, 2, 3], "expected": None})
                test_cases.append({"input": [], "expected": None})
        else:
            # Generic tests
            test_cases.append({"input": "test", "expected": None})
            test_cases.append({"input": 42, "expected": None})
            test_cases.append({"input": [1, 2, 3], "expected": None})
        
        # For generic tests where we don't know the expected output,
        # we'll need to modify the test script to just run the function
        # rather than assert equality
        
        return test_cases
    
    def _assess_code_quality(self, code):
        """Assess code quality with basic metrics"""
        quality_score = 0.5  # Start at middle
        
        # Check for docstrings
        if re.search(r'""".*?"""', code, re.DOTALL) or re.search(r"'''.*?'''", code, re.DOTALL):
            quality_score += 0.1
        
        # Check for function comments
        if re.search(r'#.*\n', code):
            quality_score += 0.05
        
        # Check for meaningful variable names (not just x, y, i, j, etc.)
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*='
        variables = re.findall(variable_pattern, code)
        
        good_var_names = [v for v in variables if len(v) > 1 and v not in ['i', 'j', 'k', 'x', 'y', 'z']]
        if len(variables) > 0:
            quality_score += 0.1 * (len(good_var_names) / len(variables))
        
        # Check for proper spacing and indentation
        if re.search(r'^\s{4}', code, re.MULTILINE) and not re.search(r'^\s{1,3}[^\s]', code, re.MULTILINE):
            quality_score += 0.05
        
        # Check for edge case handling
        if re.search(r'if.*?(?:not |len\(.*?\) == 0|== 0|is None)', code):
            quality_score += 0.1
        
        # Cap score at 1.0
        return min(1.0, quality_score)
    
    def _extract_task_concepts(self, task_text):
        """Extract concepts related to a task from the task text"""
        # Find concepts directly related to keywords in the task
        # First tokenize the task text into words
        words = re.findall(r'\b(\w+)\b', task_text.lower())
        
        # Look up each word in the concept bank
        concepts = []
        for word in words:
            if len(word) < 3:
                continue  # Skip very short words
            
            concept_id = self.model.concept_bank.find_concept_by_source(word)
            if concept_id is not None:
                concepts.append(concept_id)
        
        # Find concepts for common programming/math terms if they exist
        domain_terms = []
        if re.search(r'function|code|class|algorithm', task_text, re.IGNORECASE):
            domain_terms.extend(["function", "return", "def", "class", "algorithm"])
        
        if re.search(r'math|calculate|equation|solve', task_text, re.IGNORECASE):
            domain_terms.extend(["equation", "solve", "math", "calculation"])
        
        if re.search(r'logic|valid|reasoning', task_text, re.IGNORECASE):
            domain_terms.extend(["logic", "reasoning", "valid", "inference"])
        
        # Look up domain terms
        for term in domain_terms:
            concept_id = self.model.concept_bank.find_concept_by_source(term)
            if concept_id is not None and concept_id not in concepts:
                concepts.append(concept_id)
        
        # Limit to a reasonable number
        return concepts[:20]

    def get_evolution_stats(self):
        """Get statistics about the evolution process"""
        task_type_counts = Counter([result.task_type for result in self.task_history])
        
        return {
            "total_tasks": self.stats["total_tasks_generated"],
            "successful_tasks": self.stats["total_tasks_solved"],
            "success_rate": self.stats["total_tasks_solved"] / max(1, self.stats["total_tasks_generated"]),
            "current_complexity": self.current_complexity,
            "concepts_reinforced": self.stats["total_concepts_reinforced"],
            "task_type_distribution": dict(task_type_counts),
            "reasoning_mode_usage": dict(self.stats["reasoning_mode_usage"]),
            "focus_areas": self.focus_areas
        }
