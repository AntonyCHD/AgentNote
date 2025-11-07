import os
import nbformat as nbf
from .config import config
from .notebook_generator import NotebookGenerator
from .notebook_exporter import NotebookExporter

class NotebookManager:
    """Notebook管理器"""
    
    def __init__(self):
        self.notebook_path = config.notebook.notebook_name
    
    def create_notebook(self):
        """创建新的notebook"""
        nb = NotebookGenerator.create_notebook()
        NotebookExporter.save_notebook(nb, self.notebook_path)
        return nb
    
    def load_notebook(self):
        """加载现有的notebook"""
        if not os.path.exists(self.notebook_path):
            return self.create_notebook()
        
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            return nbf.read(f, as_version=4)
    
    def save_notebook(self, nb):
        """保存notebook"""
        NotebookExporter.save_notebook(nb, self.notebook_path)
    
    def add_markdown_cell(self, nb, markdown_text: str):
        """添加markdown cell"""
        cell = NotebookGenerator.create_markdown_cell(
            markdown_text, 
            tags=[config.notebook.markdown_cell_tag]
        )
        nb.cells.append(cell)
        self.save_notebook(nb)
        return cell
    
    def add_code_cell(self, nb, code_text: str):
        """添加代码cell"""
        cell = NotebookGenerator.create_code_cell(
            code_text, 
            tags=[config.notebook.code_cell_tag]
        )
        nb.cells.append(cell)
        self.save_notebook(nb)
        return cell
    
    def get_cell_count(self, nb):
        """获取cell数量"""
        return len(nb.cells)
    
    def cleanup_old_cells(self, nb):
        """清理旧的cell以保持数量限制"""
        if len(nb.cells) <= config.notebook.max_cells:
            return nb
        
        # 只保留最近的cell
        nb.cells = nb.cells[-config.notebook.max_cells:]
        self.save_notebook(nb)
        print(f"已清理cell，当前数量: {len(nb.cells)}")
        return nb
    
    def get_last_cell_output(self, nb):
        """获取最后一个cell的输出"""
        if not nb.cells:
            return None
        
        last_cell = nb.cells[-1]
        if last_cell.cell_type != 'code':
            return None
        
        cell_data = NotebookExporter.extract_cell_data(last_cell, len(nb.cells)-1)
        return cell_data