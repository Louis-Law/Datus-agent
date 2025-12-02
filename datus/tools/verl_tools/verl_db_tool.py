# verl_tools.py - 专门给 Verl 用的工具入口
import json
from typing import Optional, Tuple

from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool, FuncToolResult
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
except ImportError:
    raise
logger = get_logger(__name__)


class BaseVerlDBTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize DBFuncTool with configuration and schema.
        The configuration of namespace multi-database is not currently supported.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self.agent_config = load_agent_config(config=config.get("config_path"), namespace=config.get("namespace"))
        self.dialect = self.agent_config.db_type
        self.db_manager = db_manager_instance(self.agent_config.namespaces)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _tool_adapter(self, database: str) -> DBFuncTool:
        return DBFuncTool(
            connector=self.db_manager.get_conn(self.agent_config.current_namespace, database),
            agent_config=self.agent_config,
        )


class DatabaseTool(BaseVerlDBTool):
    def execute(
        self, instance_id: str, catalog: Optional[str] = "", include_sys: Optional[bool] = False
    ) -> Tuple[str, float, dict]:
        if self.agent_config.db_type == DBType.SQLITE:
            result = FuncToolResult(result=[])
        else:
            result = self._tool_adapter("").list_databases(catalog, include_sys)
        logger.debug(f"List database for {instance_id} result: {result}")
        return json.dumps(result, ensure_ascii=False), 0.0, {}


class SchemaTool(BaseVerlDBTool):
    def execute(
        self, instance_id: str, catalog: Optional[str] = "", database: Optional[str] = "", include_sys: bool = False
    ) -> Tuple[str, float, dict]:
        result = self._tool_adapter(database).list_schemas(catalog, database, include_sys)
        if result.success:
            logger.debug(f"List schemas for {instance_id} result: {len(result.result)}")
        else:
            logger.debug(f"List schemas for {instance_id} result: {result}")
        return json.dumps(result, ensure_ascii=False), 0.0, {}


class ListTableTool(BaseVerlDBTool):
    def execute(
        self,
        instance_id: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        include_views: Optional[bool] = True,
    ) -> Tuple[str, float, dict]:
        result = self._tool_adapter(database).list_tables(catalog, database, schema_name, include_views)
        if result.success:
            logger.debug(f"List tables for {instance_id} result: {len(result.result)}")
        else:
            logger.debug(f"List tables for {instance_id} result: {result}")
        return json.dumps(result, ensure_ascii=False), 0.0, {}


class DescTableTool(BaseVerlDBTool):
    def execute(
        self,
        instance_id: str,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        table_type: str = "table",
    ) -> Tuple[str, float, dict]:
        result = self._tool_adapter(database).describe_table(
            table_name, catalog, database, schema_name, table_type=table_type
        )
        logger.debug(f"Describe tables for {instance_id} result: {result}")
        return json.dumps(result, ensure_ascii=False), 0.0, {}


class QueryTool(BaseVerlDBTool):
    def execute(self, instance_id: str, sql: str, database: Optional[str] = "") -> Tuple[str, float, dict]:
        result = self._tool_adapter(database).read_query(sql)
        logger.debug(f"Describe tables for {instance_id} result: {result}")
        return json.dumps(result, ensure_ascii=False), 0.0, {}
