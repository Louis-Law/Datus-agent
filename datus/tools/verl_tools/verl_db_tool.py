# verl_tools.py - 专门给 Verl 用的工具入口
from typing import Optional

from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool, FuncToolResult
from datus.utils.constants import DBType

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
except ImportError:
    raise


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
        database_name = config.get("database") or self.agent_config.current_database
        self.tool_adapter = DBFuncTool(
            self.db_manager.get_conn(self.agent_config.current_namespace, database_name),
            self.agent_config,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema


class DatabaseTool(BaseVerlDBTool):
    def list_databases(self, catalog: Optional[str] = "", include_sys: Optional[bool] = False) -> FuncToolResult:
        if self.agent_config.db_type == DBType.SQLITE:
            return FuncToolResult(result=[])
        return self.tool_adapter.list_databases(catalog, include_sys)


class SchemaTool(BaseVerlDBTool):
    def list_schemas(
        self, catalog: Optional[str] = "", database: Optional[str] = "", include_sys: bool = False
    ) -> FuncToolResult:
        return self.tool_adapter.list_schemas(catalog, database, include_sys)


class ListTableTool(BaseVerlDBTool):
    def list_tables(
        self,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        include_views: Optional[bool] = True,
    ) -> FuncToolResult:
        return self.tool_adapter.list_tables(catalog, database, schema_name, include_views)


class DescTableTool(BaseVerlDBTool):
    def describe_table(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        table_type: str = "table",
    ) -> FuncToolResult:
        return self.tool_adapter.describe_table(table_name, catalog, database, schema_name, table_type=table_type)


class QueryTool(BaseVerlDBTool):
    def read_query(self, sql: str) -> FuncToolResult:
        return self.tool_adapter.read_query(sql)
