import time
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from .config import config
from .notebook_manager import NotebookManager
from .notebook_exporter import NotebookExporter

class NotebookExecutor:
    """Notebook执行器"""
    
    def __init__(self):
        self.manager = NotebookManager()
        self.timeout = 600  # 10分钟超时
    
    def execute_notebook(self, notebook_path: str = None):
        """执行notebook中的所有cell"""
        notebook_path = notebook_path or config.notebook.notebook_name
        
        try:
            # 读取notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbf.read(f, as_version=4)
            
            # 配置执行器
            ep = ExecutePreprocessor(timeout=self.timeout, kernel_name='python3')
            
            # 执行notebook
            ep.preprocess(nb, {'metadata': {'path': './'}})
            
            # 保存执行结果
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            
            print("Notebook 执行完成!")
            
            # 导出cell输入输出到JSON
            if config.notebook.export_json:
                NotebookExporter.export_notebook_to_json(notebook_path)
            
            return nb
            
        except Exception as e:
            print(f"执行notebook时出错: {e}")
            return None
    
    def execute_last_cell(self, nb):
        """只执行最后一个cell（新添加的cell）"""
        if not nb.cells:
            return nb
        
        # 只执行最后一个cell
        last_cell_index = len(nb.cells) - 1
        cells_to_execute = [last_cell_index]
        
        try:
            ep = ExecutePreprocessor(timeout=self.timeout, kernel_name='python3')
            
            # 创建一个只包含要执行cell的临时notebook
            temp_nb = nbf.v4.new_notebook()
            temp_nb.cells = [nb.cells[last_cell_index]]
            
            # 执行
            ep.preprocess(temp_nb, {'metadata': {'path': './'}})
            
            # 将执行结果复制回原notebook
            if temp_nb.cells:
                nb.cells[last_cell_index] = temp_nb.cells[0]
            
            # 保存
            self.manager.save_notebook(nb)
            print("最后一个cell执行完成!")
            
            return nb
            
        except Exception as e:
            print(f"执行最后一个cell时出错: {e}")
            return nb