# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create an interaction instance.

        Args:
            instance_id: The instance id of the interaction.

        Returns:
            The instance id of the interaction.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Generate a response for the current turn of interaction.

        Returns a tuple containing:
        - should_terminate (bool): True if the interaction sequence should end.
        - response_content (str): The textual content of the response.
        - current_turn_score (float): The score for this specific turn/response.
        - additional_data (dict): Any extra information or metadata.
        """
        should_terminate: bool = False
        response_content: str = "Your current result seems acceptable."
        current_turn_score: float = 0.0
        additional_data: dict[str, Any] = {}
        return should_terminate, response_content, current_turn_score, additional_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate a score for the current interaction state."""
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction session and release resources."""
        pass
