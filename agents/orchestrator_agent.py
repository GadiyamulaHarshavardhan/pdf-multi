from .autonomous_agent import AutonomousAgent
from typing import Dict, Any, List
import asyncio
from datetime import datetime
from memory.state_manager import global_state_manager

class OrchestratorAgent(AutonomousAgent):
    def __init__(self, llm, agents: Dict[str, AutonomousAgent]):
        super().__init__(
            name="orchestrator_agent",
            role="Workflow Orchestrator",
            goal="Coordinate multi-agent workflow for web scraping and PDF organization"
        )
        self.llm = llm
        self.agents = agents
        self.state_manager = global_state_manager
    
    async def _analyze_problem(self, context: Dict) -> Dict:
        """Analyze the multi-agent workflow requirements"""
        target_url = context.get('target_url', 'Unknown')
        max_depth = context.get('max_depth', 2)
        
        return {
            "task": "multi_agent_web_scraping",
            "target_url": target_url,
            "max_depth": max_depth,
            "workflow": ["crawl_website", "find_pdfs", "download_pdfs", "organize_files"],
            "agents_involved": list(self.agents.keys()),
            "coordination_strategy": "sequential_with_state_sharing",
            "error_handling": "graceful_degradation_with_fallbacks"
        }
    
    async def _create_execution_plan(self, analysis: Dict) -> Dict:
        """Create execution plan for multi-agent workflow"""
        return {
            "workflow_steps": [
                {
                    "step": 1,
                    "agent": "crawler_agent",
                    "task": "crawl_website_for_pdfs",
                    "input": {"base_url": analysis["target_url"], "max_depth": analysis["max_depth"]},
                    "output_key": "found_pdfs"
                },
                {
                    "step": 2,
                    "agent": "pdf_organizer_agent", 
                    "task": "download_and_organize_pdfs",
                    "input": {"pdf_urls": "state:found_pdfs", "base_url": analysis["target_url"]},
                    "output_key": "organized_pdfs"
                }
            ],
            "parallel_execution": False,
            "state_sharing": True,
            "error_recovery": True,
            "monitoring": True
        }
    
    async def _self_check_plan(self, plan: Dict) -> Dict:
        """Validate the workflow plan"""
        issues = []
        
        if len(plan["workflow_steps"]) == 0:
            issues.append("No workflow steps defined")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Ensure all agents are available", "Verify state sharing mechanisms"]
        }
    
    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute the multi-agent workflow"""
        results = {}
        workflow_steps = plan["workflow_steps"]
        
        for step_info in workflow_steps:
            step_num = step_info["step"]
            agent_name = step_info["agent"]
            task = step_info["task"]
            input_data = step_info["input"]
            
            print(f"🔄 Executing step {step_num}: {task} with {agent_name}")
            
            # Resolve any state references in input data
            resolved_input = await self._resolve_input_references(input_data)
            
            # Get the agent
            agent = self.agents.get(agent_name)
            if not agent:
                print(f"❌ Agent {agent_name} not found")
                continue
            
            try:
                # Execute the agent with the resolved input
                agent_result = await agent.execute(resolved_input)
                
                # Store result in state manager using the output key
                output_key = step_info.get("output_key")
                if output_key:
                    self.state_manager.store_state(output_key, agent_result)
                    results[output_key] = agent_result
                
                print(f"✅ Step {step_num} completed: {task}")
                
            except Exception as e:
                print(f"❌ Step {step_num} failed: {task} - {str(e)}")
                # Log error but continue with workflow
                results[f"{output_key}_error"] = str(e)
                
                # Check if this step is critical for subsequent steps
                if step_info.get("critical", True):
                    print("⚠️ Critical step failed, continuing with available data...")
        
        return {
            "workflow_results": results,
            "completed_steps": len([s for s in workflow_steps if s.get("output_key") in results]),
            "total_steps": len(workflow_steps),
            "execution_summary": f"Executed {len(workflow_steps)} workflow steps"
        }
    
    async def _verify_result(self, result: Dict) -> Dict:
        """Verify workflow execution results"""
        issues = []
        
        workflow_results = result.get("workflow_results", {})
        if not workflow_results:
            issues.append("No workflow results generated")
        
        # Check for errors in results
        errors = [k for k in workflow_results.keys() if "_error" in k]
        if errors:
            issues.extend([f"Step error: {err}" for err in errors])
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "workflow_results": workflow_results,
            "suggestions": ["Check agent configurations", "Verify input data validity"]
        }
    
    async def _debug_and_adapt(self, plan: Dict, error: Exception):
        """Debug and adapt workflow based on errors"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            # Reduce parallelism or add more delays
            plan["parallel_execution"] = False
            self.log_action("adapt_workflow", {"action": "disable_parallel_execution"}, "timeout_error")
        
        if "memory" in error_str or "oom" in error_str:
            # Enable memory optimization
            plan["memory_optimization"] = True
            self.log_action("adapt_workflow", {"action": "enable_memory_optimization"}, "memory_error")
    
    async def _resolve_input_references(self, input_data: Dict) -> Dict:
        """Resolve state references in input data"""
        resolved = {}
        
        for key, value in input_data.items():
            if isinstance(value, str) and value.startswith("state:"):
                # Extract state key
                state_key = value[6:]  # Remove "state:" prefix
                state_value = self.state_manager.retrieve_state(state_key)
                resolved[key] = state_value or []
            else:
                resolved[key] = value
        
        return resolved
    
    async def execute(self, state: Dict) -> Dict:
        """Main execution method"""
        # Use the autonomous workflow from parent class
        return await super().execute(state)
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return {
            "agents": list(self.agents.keys()),
            "state_manager_usage": self.state_manager.get_memory_usage(),
            "recent_states": self.state_manager.get_recent_states(5),
            "workflow_history": self.state_manager.get_workflow_history(10)
        }