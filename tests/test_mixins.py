import json
import pytest
from django.http import JsonResponse
from unittest.mock import MagicMock
from model_bakery import baker

from django_chain.models import Workflow
from django_chain.mixins import (
    JSONResponseMixin,
    ModelRetrieveMixin,
    ModelListMixin,
    ModelCreateMixin,
    ModelUpdateMixin,
    ModelDeleteMixin,
    ModelActivateDeactivateMixin,
)


@pytest.mark.django_db
class TestMixins:
    """Consolidated mixin tests."""

    def test_json_response_mixin(self):
        """Test JSONResponseMixin functionality."""
        view = JSONResponseMixin()

        response = view.render_json_response(data={"hello": "world"})
        assert isinstance(response, JsonResponse)
        assert response.status_code == 200
        assert response.content == b'{"hello": "world"}'

        response = view.render_json_response(data={"error": "test"}, status=400)
        assert response.status_code == 400

        error_response = view.json_error_response("Test error", status=500)
        assert error_response.status_code == 500
        data = json.loads(error_response.content)
        assert data["error"] == "Test error"

    def test_model_retrieve_mixin(self):
        """Test ModelRetrieveMixin methods."""
        mixin = ModelRetrieveMixin()
        mixin.model_class = Workflow
        mixin.serializer_method = MagicMock()

        workflow = baker.make(
            Workflow,
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
            ],
        )
        result = mixin.get_object(str(workflow.id))
        assert result == workflow

    def test_model_list_mixin(self):
        """Test ModelListMixin functionality."""
        mixin = ModelListMixin()
        mixin.model_class = Workflow
        mixin.serializer_method = MagicMock()

        workflows = baker.make(
            Workflow,
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
            ],
            _quantity=3,
        )

        queryset = mixin.get_queryset(MagicMock())
        assert queryset.count() >= 3

    def test_model_crud_mixins(self):
        """Test CRUD mixin functionality."""
        workflow = baker.make(
            Workflow,
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
            ],
        )

        create_mixin = ModelCreateMixin()
        create_mixin.model_class = Workflow
        create_mixin.required_fields = ["name"]

        update_mixin = ModelUpdateMixin()
        update_mixin.model_class = Workflow

        delete_mixin = ModelDeleteMixin()
        delete_mixin.model_class = Workflow

        assert hasattr(create_mixin, "create_object")
        assert hasattr(update_mixin, "update_object")
        assert hasattr(delete_mixin, "delete_object")

    def test_model_activate_deactivate_mixin(self):
        """Test ModelActivateDeactivateMixin functionality."""
        workflow = baker.make(
            Workflow,
            id=1,
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
            ],
        )

        mixin = ModelActivateDeactivateMixin()
        mixin.model_class = Workflow
        mixin.serializer_method = MagicMock()

        mixin.post(workflow, 1, action="activate")
        workflow.refresh_from_db()
        assert workflow.is_active

        mixin.post(workflow, 1, action="deactivate")
        workflow.refresh_from_db()
        assert not workflow.is_active

    def test_mixin_error_handling(self):
        """Test error handling in mixins."""
        mixin = ModelRetrieveMixin()
        mixin.model_class = Workflow
        mixin.serializer_method = MagicMock()
        result = mixin.get_object("nonexistent-id")
        assert result is None
