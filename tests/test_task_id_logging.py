"""
测试 task_id 是否正确记录到日志中
"""
import unittest
from unittest.mock import MagicMock, patch
from rag_assistant.agent import run_agent


class TestTaskIdLogging(unittest.TestCase):
    
    def test_task_id_in_record(self):
        """验证 task_id 出现在 record 中"""
        # 准备 mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.__enter__.return_value = MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="图书馆在主楼二层。", tool_calls=None),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = [[0.1] * 512]
        
        knowledge_chunks = [{"chunk_id": 1, "content": "图书馆位于主楼二层"}]
        chunk_vectors = [[0.1] * 512]
        
        # 调用 run_agent，传入 task_id=42
        record = run_agent(
            question="图书馆在哪里？",
            client=mock_client,
            knowledge_chunks=knowledge_chunks,
            chunk_vectors=chunk_vectors,
            embedding_model=mock_embedding_model,
            task_id=42,  # ← 传入 task_id
        )
        
        # 验证 task_id 在 record 中
        self.assertIn("task_id", record)
        self.assertEqual(record["task_id"], 42)
    
    def test_task_id_none(self):
        """验证不传 task_id 时为 None"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.__enter__.return_value = MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="图书馆在主楼二层。", tool_calls=None),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = [[0.1] * 512]
        
        knowledge_chunks = [{"chunk_id": 1, "content": "图书馆位于主楼二层"}]
        chunk_vectors = [[0.1] * 512]
        
        # 不传 task_id
        record = run_agent(
            question="图书馆在哪里？",
            client=mock_client,
            knowledge_chunks=knowledge_chunks,
            chunk_vectors=chunk_vectors,
            embedding_model=mock_embedding_model,
        )
        
        # task_id 应该是 None
        self.assertIn("task_id", record)
        self.assertIsNone(record["task_id"])


if __name__ == "__main__":
    unittest.main()
