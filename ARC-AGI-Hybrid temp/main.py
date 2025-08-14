"""
Main entry point for Hybrid ARC Agent
Combines Tree-of-Thought symbolic reasoning with neural program synthesis.
This version fixes:
- time-limit handling
- removal of hidden global args in evaluate
- safe matplotlib usage in headless envs
- robust error handling and logging
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure local package resolvable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # safe for headless
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from agent.HybridAgent import HybridAgent  # package layout
except Exception:
    from agent.HybridAgent import HybridAgent  # flat layout fallback


# ---------- Logging ----------

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )



# ---------- Data loading ----------

def load_arc_tasks(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load ARC tasks from a directory of *.json files.
    Each file is expected to follow ARC train/test pairs structure.
    """
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")
    tasks: List[Dict[str, Any]] = []
    for f in sorted(p.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                task = json.load(fh)
            task['__filename__'] = f.name
            tasks.append(task)
        except Exception as e:
            logging.warning(f"Failed to load {f.name}: {e}")
    logging.info(f"Loaded {len(tasks)} tasks from {data_dir}")
    return tasks


# ---------- Evaluation ----------

@torch.no_grad()
def evaluate_agent(agent: HybridAgent, tasks: List[Dict[str, Any]],
                   output_dir: str, time_limit_s: Optional[float] = None,
                   debug: bool = False) -> Dict[str, Any]:
    """
    Evaluate the agent on provided tasks.
    time_limit_s: overall wall-clock budget; None for no limit.
    """
    results: Dict[str, Any] = {
        "num_tasks": len(tasks),
        "num_success": 0,
        "details": []
    }

    overall_start = time.time()

    for ti, task in enumerate(tasks):
        if time_limit_s is not None and (time.time() - overall_start) > time_limit_s:
            logging.info("Overall time limit reached — stopping evaluation.")
            break

        fname = task.get('__filename__', f"task_{ti}.json")
        train = task.get('train', [])
        test = task.get('test', [])

        if not train or not test:
            logging.error(f"Task {fname} missing train/test — skipping.")
            continue

        # Convert to tensors
        def to_tensor(grid):
            return torch.tensor(grid, dtype=torch.long)

        task_success = 0
        task_details: List[Dict[str, Any]] = []

        for pi, pair in enumerate(test):
            try:
                test_input = to_tensor(pair['input'])
                target_output = to_tensor(pair.get('output')) if 'output' in pair else None

                train_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = [
                    (to_tensor(t['input']), to_tensor(t['output'])) for t in train
                ]

                out = agent.solve_task(
                    train_pairs=train_pairs,
                    test_input=test_input,
                    target_output=target_output,
                    beam_width=5,
                    exec_time_budget_s=5.0,  # per-pair budget (tune)
                )

                pred = out["prediction"]
                program = out["program"]
                score = out["score"]

                detail = {
                    "task": fname,
                    "pair_index": pi,
                    "program": program,
                    "score": score,
                }

                if target_output is not None and pred is not None and torch.equal(pred, target_output):
                    task_success += 1
                    detail["success"] = True
                else:
                    detail["success"] = False

                task_details.append(detail)

                if debug:
                    logging.debug(f"[{fname} pair {pi}] score={score:.4f} success={detail['success']}")
            except Exception as e:
                logging.error(f"Error processing pair {pi} in task {fname}: {e}")
                continue

        results["num_success"] += task_success
        results["details"].extend(task_details)

    return results


# ---------- Demo ----------

def demo_agent(agent: HybridAgent, task_file: str, output_dir: str) -> None:
    with open(task_file, "r", encoding="utf-8") as fh:
        task = json.load(fh)
    tasks = [task]
    results = evaluate_agent(agent, tasks, output_dir, time_limit_s=None, debug=True)
    logging.info(f"Demo results: {results}")


# ---------- CLI ----------

def make_agent_from_args(args: argparse.Namespace) -> HybridAgent:
    # Minimal config; replace with your structured config if available
    config: Dict[str, Any] = {
        "vit": {
            "num_colours": 10,
            "emb_dim": 256,
            "depth": 6,
            "num_heads": 8,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "max_grid_size": 64,
        },
        "program_synthesiser": {
            "emb_dim": 256,
            "num_heads": 8,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "depth": 2,
            "d_model": 256,
            "vocab_size": 128,
            "max_steps": 16,
        }
    }
    return HybridAgent(config=config, seed=args.seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\ARC-AGI-Hybrid temp\ARC-AGI-Hybrid temp\Data\training",
        help="Path to ARC JSON task directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file path for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Create agent from config
    agent_instance = make_agent_from_args(args)

    # Load ARC tasks
    tasks = load_arc_tasks(args.data_dir)

    # Run evaluation
    results = evaluate_agent(
        agent=agent_instance,
        tasks=tasks,
        output_dir=os.path.dirname(args.output),
        debug=args.verbose
    )

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved results to {args.output}")
    logging.info(f"Evaluation complete: {results['num_success']}/{results['num_tasks']} successful tasks")


if __name__ == "__main__":
    main()








"""
Main entry point for Hybrid ARC Agent
Combines Tree-of-Thought symbolic reasoning with neural program synthesis
"""
'''
import argparse
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.HybridAgent import HybridAgent
from agent.utils import (
    load_arc_task, save_results, setup_logging, log_task_results,
    compute_metrics, PerformanceMonitor, plot_task, plot_comparison
)
from config import DEFAULT_CONFIG

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hybrid ARC Agent")
    
    # Mode selection
    parser.add_argument("--mode", choices=["evaluate", "train", "demo"], 
                       default="evaluate", help="Operation mode")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="data/evaluation",
                       help="Directory containing ARC tasks")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory for model checkpoints")
    
    # Model configuration
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Path to checkpoint to load")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                       help="Path to save checkpoint")
    
    # Evaluation settings
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="Maximum number of tasks to evaluate")
    parser.add_argument("--time_limit", type=int, default=3600,
                       help="Time limit in seconds for evaluation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    
    # Logging
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    return parser.parse_args()

def load_tasks(data_dir: str, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load ARC tasks from directory."""
    tasks = []
    
    if not os.path.exists(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return tasks
    
    # Find all JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if max_tasks:
        json_files = json_files[:max_tasks]
    
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        try:
            task = load_arc_task(filepath)
            task['filename'] = filename
            tasks.append(task)
            logging.info(f"Loaded task: {filename}")
        except Exception as e:
            logging.warning(f"Failed to load {filename}: {e}")
    
    logging.info(f"Loaded {len(tasks)} tasks from {data_dir}")
    return tasks

def evaluate_agent(agent: HybridAgent, tasks: List[Dict[str, Any]], 
                  output_dir: str, time_limit: int) -> Dict[str, Any]:
    """Evaluate agent on tasks."""
    logging.info(f"Starting evaluation on {len(tasks)} tasks")
    
    # Setup monitoring
    monitor = PerformanceMonitor()
    predictions = []
    targets = []
    methods = []
    results = []
    
    start_time = monitor.start_time
    
    for i, task in enumerate(tasks):
        if time_limit and monitor.start_time and start_time:
            if (monitor.start_time - start_time).total_seconds() > time_limit:
                logging.warning(f"Time limit reached after {i} tasks")  
                break
        
        monitor.start_task()
        
        try:
            # Get test pairs
            test_pairs = task.get('test', [])
            if not test_pairs:
                logging.warning(f"No test pairs for task {task.get('filename', 'unknown')}")
                continue
            
            # Evaluate on each test pair
            for j, (test_input, test_output) in enumerate(test_pairs):
                # Solve task
                prediction, method_info = agent.solve_task(
                    task['train'], test_input
                )
                
                # Compute accuracy
                accuracy = 1.0 - (prediction != test_output).float().mean().item()
                success = accuracy > 0.9  # 90% threshold
                
                # Store results
                predictions.append(prediction)
                targets.append(test_output)
                methods.append(method_info['method'])
                
                # Log results
                log_task_results(
                    task.get('filename', f'task_{i}'),
                    method_info['method'],
                    accuracy,
                    method_info.get('confidence', 0.0),
                    method_info.get('steps'),
                    method_info.get('program')
                )
                
                # Store detailed results
                result = {
                    'task_id': task.get('filename', f'task_{i}'),
                    'test_pair': j,
                    'method': method_info['method'],
                    'accuracy': accuracy,
                    'confidence': method_info.get('confidence', 0.0),
                    'success': success,
                    'method_info': method_info
                }
                results.append(result)
                
                monitor.end_task(method_info['method'], success)
                
                # Debug visualization
                if args.debug and i < 5:  # Only for first 5 tasks
                    fig = plot_comparison(test_input, prediction, test_output)
                    fig.savefig(os.path.join(output_dir, f'debug_task_{i}_test_{j}.png'))
                    plt.close(fig)
        
        except Exception as e:
            logging.error(f"Error evaluating task {task.get('filename', 'unknown')}: {e}")
            monitor.end_task('error', False)
    
    # Compute final metrics
    metrics = compute_metrics(predictions, targets, methods)
    performance_summary = monitor.get_summary()
    
    # Combine results
    final_results = {
        'metrics': metrics,
        'performance': performance_summary,
        'detailed_results': results,
        'agent_stats': agent.get_stats()
    }
    
    return final_results

def train_agent(agent: HybridAgent, tasks: List[Dict[str, Any]], 
                epochs: int, learning_rate: float, batch_size: int):
    """Train the agent on tasks."""
    logging.info(f"Starting training on {len(tasks)} tasks for {epochs} epochs")
    
    # Prepare training data
    train_data = []
    for task in tasks:
        train_pairs = task['train']
        test_pairs = task.get('test', [])
        
        # Use test pairs as training targets if available
        if test_pairs:
            for test_input, test_output in test_pairs:
                train_data.append((train_pairs, test_input, test_output))
        else:
            # Use last training pair as target
            if len(train_pairs) > 1:
                train_input, train_output = train_pairs[-1]
                train_data.append((train_pairs[:-1], train_input, train_output))
    
    # Train agent
    agent.train(train_data, epochs=epochs, learning_rate=learning_rate)
    
    logging.info("Training completed")

def demo_agent(agent: HybridAgent, task_file: str, output_dir: str):
    """Run demo on a single task."""
    logging.info(f"Running demo on task: {task_file}")
    
    # Load task
    task = load_arc_task(task_file)
    
    # Get test pairs
    test_pairs = task.get('test', [])
    if not test_pairs:
        logging.error("No test pairs found in task")
        return
    
    # Run predictions
    for i, (test_input, test_output) in enumerate(test_pairs):
        # Solve task
        prediction, method_info = agent.solve_task(task['train'], test_input)
        
        # Compute accuracy
        accuracy = 1.0 - (prediction != test_output).float().mean().item()
        
        # Log results
        log_task_results(
            task.get('task_id', 'demo_task'),
            method_info['method'],
            accuracy,
            method_info.get('confidence', 0.0),
            method_info.get('steps'),
            method_info.get('program')
        )
        
        # Create visualization
        fig = plot_task(
            task['train'], 
            test_input, 
            test_output, 
            prediction
        )
        
        # Save visualization
        output_file = os.path.join(output_dir, f'demo_result_{i}.png')
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Demo result saved to: {output_file}")
        logging.info(f"Accuracy: {accuracy:.3f}, Method: {method_info['method']}")

def main():
    """Main function."""
    global args
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logging.info("Starting Hybrid ARC Agent")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load configuration
    config = DEFAULT_CONFIG.__dict__
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                user_config = json.load(f)
            config.update(user_config)
            logging.info(f"Loaded configuration from: {args.config_file}")
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
    
    # Initialize agent
    logging.info("Initializing Hybrid Agent")
    agent = HybridAgent(config)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        try:
            agent.load_agent(args.load_checkpoint)
            logging.info(f"Loaded checkpoint from: {args.load_checkpoint}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
    
    # Run based on mode
    if args.mode == "evaluate":
        # Load tasks
        tasks = load_tasks(args.data_dir, args.max_tasks)
        if not tasks:
            logging.error("No tasks found for evaluation")
            return
        
        # Evaluate agent
        results = evaluate_agent(agent, tasks, args.output_dir, args.time_limit)
        
        # Save results
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        save_results(results, results_file)
        logging.info(f"Results saved to: {results_file}")
        
        # Print summary
        metrics = results['metrics']
        performance = results['performance']
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Tasks: {performance['total_tasks']}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}")
        print(f"Success Rate: {performance['success_rate']:.3f}")
        print(f"Average Time per Task: {performance['avg_time_per_task']:.2f}s")
        
        print("\nMethod Breakdown:")
        for method, data in metrics['method_breakdown'].items():
            print(f"  {method}: {data['count']} tasks, {data['avg_accuracy']:.3f} accuracy")
        
        print("\nAgent Statistics:")
        agent_stats = results['agent_stats']
        print(f"  Symbolic Success Rate: {agent_stats.get('symbolic_success_rate', 0):.3f}")
        print(f"  Neural Success Rate: {agent_stats.get('neural_success_rate', 0):.3f}")
        print(f"  Hybrid Success Rate: {agent_stats.get('hybrid_success_rate', 0):.3f}")
    
    elif args.mode == "train":
        # Load training tasks
        tasks = load_tasks(args.data_dir, args.max_tasks)
        if not tasks:
            logging.error("No tasks found for training")
            return
        
        # Train agent
        train_agent(agent, tasks, args.epochs, args.learning_rate, args.batch_size)
        
        # Save checkpoint
        if args.save_checkpoint:
            checkpoint_file = os.path.join(args.checkpoint_dir, args.save_checkpoint)
            agent.save_agent(checkpoint_file)
            logging.info(f"Checkpoint saved to: {checkpoint_file}")
    
    elif args.mode == "demo":
        # Run demo on single task
        if not os.path.exists(args.data_dir):
            logging.error(f"Task file not found: {args.data_dir}")
            return
        
        demo_agent(agent, args.data_dir, args.output_dir)
    
    logging.info("Hybrid ARC Agent completed")

if __name__ == "__main__":
    main() 
'''