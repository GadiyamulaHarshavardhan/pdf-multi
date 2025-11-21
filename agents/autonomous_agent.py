"""
Base class for autonomous agents with reasoning capabilities
"""

import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any

class AutonomousAgent(ABC):
    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup agent-specific logger"""
        logger = logging.getLogger(self.name)
        handler = logging.FileHandler(f'logs/{self.name}.jsonl')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def log_action(self, action: str, data: Dict, status: str, error: str = None):
        """Log agent actions"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "action": action,
            "data": data,
            "status": status,
            "error": error
        }
        self.logger.info(json.dumps(log_entry))
    
    @abstractmethod
    async def _analyze_problem(self, context: Dict) -> Dict:
        """Analyze the problem/task"""
        pass
    
    @abstractmethod
    async def _create_execution_plan(self, analysis: Dict) -> Dict:
        """Create execution plan"""
        pass
    
    @abstractmethod
    async def _self_check_plan(self, plan: Dict) -> Dict:
        """Validate the plan"""
        pass
    
    @abstractmethod
    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute the plan"""
        pass
    
    @abstractmethod
    async def _verify_result(self, result: Dict) -> Dict:
        """Verify execution results"""
        pass
    
    @abstractmethod
    async def _debug_and_adapt(self, plan: Dict, error: Exception):
        """Debug and adapt strategy"""
        pass
    
    async def execute(self, context: Dict) -> Dict:
        """Main execution method following autonomous workflow"""
        try:
            # Step 1: Analyze the problem
            self.log_action("analyze_problem", context, "started")
            analysis = await self._analyze_problem(context)
            self.log_action("analyze_problem", analysis, "completed")
            
            # Step 2: Create execution plan
            self.log_action("create_plan", analysis, "started")
            plan = await self._create_execution_plan(analysis)
            self.log_action("create_plan", plan, "completed")
            
            # Step 3: Self-check the plan
            self.log_action("self_check", plan, "started")
            validation = await self._self_check_plan(plan)
            self.log_action("self_check", validation, "completed")
            
            if not validation["valid"]:
                return {
                    "status": "plan_validation_failed",
                    "analysis": analysis,
                    "plan": plan,
                    "validation_issues": validation["issues"],
                    "suggestions": validation["suggestions"]
                }
            
            # Step 4: Execute the plan
            self.log_action("execute_plan", plan, "started")
            result = await self._execute_plan(plan)
            self.log_action("execute_plan", result, "completed")
            
            # Step 5: Verify results
            self.log_action("verify_result", result, "started")
            verification = await self._verify_result(result)
            self.log_action("verify_result", verification, "completed")
            
            if not verification["success"]:
                return {
                    "status": "verification_failed",
                    "analysis": analysis,
                    "plan": plan,
                    "result": result,
                    "verification_issues": verification["issues"],
                    "suggestions": verification["suggestions"]
                }
            
            return {
                "status": "success",
                "analysis": analysis,
                "plan": plan,
                "result": result,
                "verification": verification
            }
            
        except Exception as e:
            self.log_action("execute", context, "failed", str(e))
            
            # Try to debug and adapt
            try:
                await self._debug_and_adapt(plan if 'plan' in locals() else {}, e)
            except Exception as debug_error:
                self.log_action("debug_adapt", {}, "failed", str(debug_error))
            
            return {
                "status": "error",
                "error": str(e),
                "agent": self.name
            }