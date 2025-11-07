import time
import yaml
import os
from typing import List, Dict, Any, Optional
from ..core.config import config
from ..core.deepseek_client import DeepSeekClient
from ..core.content_parser import ContentParser
from ..core.notebook_manager import NotebookManager
from ..core.executor import NotebookExecutor
from ..core.notebook_exporter import NotebookExporter

class NoteAgent:
    """NoteAgent智能体 - 自动化任务执行和Notebook生成"""
    
    def __init__(self, api_key: str = None):
        # 初始化组件
        self.client = DeepSeekClient(api_key)
        self.parser = ContentParser()
        self.manager = NotebookManager()
        self.executor = NotebookExecutor()
        self.exporter = NotebookExporter()
        
        # 加载提示词
        self.prompts = self._load_prompts()
        
        # 状态跟踪
        self.current_task = None
        self.execution_plan = []
        self.current_step = 0
        self.execution_history = []
    
    def _load_prompts(self) -> Dict[str, Any]:
        """加载提示词模板"""
        prompts_path = os.path.join(os.path.dirname(__file__), '../prompts/prompts.yaml')
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载提示词失败: {e}")
            return {}
    
    def _get_prompt(self, category: str, key: str, **kwargs) -> str:
        """获取格式化后的提示词"""
        if category in self.prompts and key in self.prompts[category]:
            template = self.prompts[category][key]
            return template.format(**kwargs)
        return ""
    
    def plan_task(self, task_description: str) -> List[Dict[str, str]]:
        """任务规划"""
        print(f"开始规划任务: {task_description}")
        
        system_prompt = self._get_prompt('system_prompts', 'planner')
        user_prompt = self._get_prompt('task_prompts', 'planning', 
                                     task_description=task_description)
        
        plan_content = self.client.generate_with_retry(system_prompt, user_prompt)
        if not plan_content:
            print("任务规划失败")
            return []
        
        # 解析规划步骤
        steps = self._parse_planning_steps(plan_content)
        self.execution_plan = steps
        self.current_task = task_description
        self.current_step = 0
        
        print(f"任务规划完成，共 {len(steps)} 个步骤")
        return steps
    
    def _parse_planning_steps(self, plan_content: str) -> List[Dict[str, str]]:
        """解析规划步骤"""
        steps = []
        lines = plan_content.split('\n')
        
        current_step = None
        for line in lines:
            line = line.strip()
            if line.startswith('### 步骤'):
                if current_step:
                    steps.append(current_step)
                # 提取步骤信息
                step_parts = line.split(':', 1)
                step_name = step_parts[1].strip() if len(step_parts) > 1 else "未命名步骤"
                current_step = {
                    'name': step_name,
                    'description': '',
                    'expected_output': ''
                }
            elif line.startswith('- **描述**:') and current_step:
                current_step['description'] = line.replace('- **描述**:', '').strip()
            elif line.startswith('- **预期输出**:') and current_step:
                current_step['expected_output'] = line.replace('- **预期输出**:', '').strip()
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def execute_step(self, step_index: int) -> bool:
        """执行单个步骤"""
        if step_index >= len(self.execution_plan):
            print("步骤索引超出范围")
            return False
        
        step = self.execution_plan[step_index]
        print(f"执行步骤 {step_index + 1}: {step['name']}")
        
        # 加载当前notebook
        nb = self.manager.load_notebook()
        
        # 添加上下文信息
        context = self._build_context(step_index)
        
        # 生成步骤说明
        self._add_step_description(nb, step, step_index)
        
        # 生成和执行代码
        success = self._generate_and_execute_code(nb, step, context, step_index)
        
        if success:
            self.current_step = step_index + 1
            self.execution_history.append({
                'step': step_index,
                'name': step['name'],
                'status': 'success'
            })
        else:
            self.execution_history.append({
                'step': step_index,
                'name': step['name'],
                'status': 'failed'
            })
        
        return success
    
    def _build_context(self, step_index: int) -> str:
        """构建上下文信息"""
        context = f"任务: {self.current_task}\n"
        context += f"当前步骤: {step_index + 1}/{len(self.execution_plan)}\n"
        context += f"步骤名称: {self.execution_plan[step_index]['name']}\n"
        
        if step_index > 0:
            context += "已完成步骤:\n"
            for i in range(step_index):
                context += f"- {self.execution_plan[i]['name']}\n"
        
        return context
    
    def _add_step_description(self, nb, step: Dict[str, str], step_index: int):
        """添加步骤描述到notebook"""
        markdown_content = f"## 步骤 {step_index + 1}: {step['name']}\n\n"
        markdown_content += f"**描述**: {step['description']}\n\n"
        markdown_content += f"**预期输出**: {step['expected_output']}\n\n"
        markdown_content += f"**执行时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        self.manager.add_markdown_cell(nb, markdown_content)
    
    def _generate_and_execute_code(self, nb, step: Dict[str, str], context: str, step_index: int) -> bool:
        """生成和执行代码"""
        max_retries = config.agent.max_retries
        
        for attempt in range(max_retries):
            print(f"生成代码 (尝试 {attempt + 1}/{max_retries})...")
            
            # 生成代码
            code_success, markdown_content, python_code = self._generate_code(step, context, attempt)
            if not code_success:
                continue
            
            # 添加生成的代码到notebook
            if markdown_content:
                self.manager.add_markdown_cell(nb, markdown_content)
            
            if python_code:
                code_cell = self.manager.add_code_cell(nb, python_code)
                
                # 执行代码
                if config.agent.enable_execution:
                    execution_success = self._execute_and_verify(nb, step_index, attempt)
                    if execution_success:
                        return True
                else:
                    return True  # 如果不执行代码，直接返回成功
        
        print(f"步骤 {step_index + 1} 执行失败，已达到最大重试次数")
        return False
    
    def _generate_code(self, step: Dict[str, str], context: str, attempt: int) -> tuple:
        """生成代码内容"""
        system_prompt = self._get_prompt('system_prompts', 'code_generator')
        user_prompt = self._get_prompt('task_prompts', 'code_generation',
                                     step_description=step['description'],
                                     context=context)
        
        # 如果是重试，添加错误信息
        if attempt > 0:
            last_error = self.execution_history[-1].get('last_error', '未知错误')
            user_prompt += f"\n\n之前的执行错误: {last_error}\n请修复这个错误。"
        
        content = self.client.generate_with_retry(system_prompt, user_prompt)
        if not content:
            return False, "", ""
        
        # 解析内容
        python_code, markdown_content = self.parser.extract_python_code(content)
        
        # 验证代码语法
        if python_code:
            is_valid, validation_msg = self.parser.validate_python_code(python_code)
            print(f"代码验证: {validation_msg}")
            if not is_valid:
                print("代码语法有问题，需要重新生成")
                return False, "", ""
        
        return True, markdown_content, python_code
    
    def _execute_and_verify(self, nb, step_index: int, attempt: int) -> bool:
        """执行代码并验证结果"""
        print("执行代码...")
        
        # 执行最后一个cell（新添加的代码cell）
        nb = self.executor.execute_last_cell(nb)
        
        # 获取执行结果
        last_cell_output = self.manager.get_last_cell_output(nb)
        
        # 检查执行错误
        if last_cell_output and self.parser.contains_execution_errors(last_cell_output):
            error_message = self.parser.get_error_message(last_cell_output)
            print(f"代码执行错误: {error_message}")
            
            # 记录错误信息用于重试
            if self.execution_history:
                self.execution_history[-1]['last_error'] = error_message
            
            return False
        
        print("代码执行成功")
        return True
    
    def run_task(self, task_description: str) -> bool:
        """运行完整任务"""
        print(f"开始执行任务: {task_description}")
        
        # 任务规划
        steps = self.plan_task(task_description)
        if not steps:
            print("任务规划失败")
            return False
        
        # 初始化notebook
        nb = self.manager.load_notebook()
        self.manager.add_markdown_cell(nb, f"# 任务: {task_description}\n\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 按顺序执行步骤
        for i in range(len(steps)):
            success = self.execute_step(i)
            if not success and config.agent.enable_auto_fix:
                print(f"步骤 {i + 1} 执行失败，尝试自动修复...")
                # 这里可以添加更复杂的修复逻辑
                success = self.execute_step(i)  # 重试一次
            
            if not success:
                print(f"任务在步骤 {i + 1} 失败")
                return False
            
            # 清理旧cell
            nb = self.manager.cleanup_old_cells(nb)
            
            # 等待间隔
            time.sleep(config.notebook.sleep_interval)
        
        # 添加任务完成标记
        nb = self.manager.load_notebook()
        self.manager.add_markdown_cell(nb, f"## 任务完成\n\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n所有步骤执行完毕!")
        
        print("任务执行完成!")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'current_task': self.current_task,
            'total_steps': len(self.execution_plan),
            'current_step': self.current_step,
            'execution_history': self.execution_history,
            'completion_percentage': (self.current_step / len(self.execution_plan)) * 100 if self.execution_plan else 0
        }