import json
import pytest
from django.http import HttpRequest, JsonResponse
from model_bakery import baker
from django.views import View
from django.core.exceptions import ValidationError
from django.test import RequestFactory
from unittest.mock import MagicMock

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


def test_render_json_response():
    view = JSONResponseMixin()
    response = view.render_json_response(data={"hello": "world"})
    assert isinstance(response, JsonResponse)
    assert response.status_code == 200
    assert response.content == b'{"hello": "world"}'


@pytest.mark.parametrize(
    "error_input,expected",
    [("Invalid", {"error": "Invalid"}), ({"field": "required"}, {"errors": {"field": "required"}})],
)
def test_json_error_response(error_input, expected):
    view = JSONResponseMixin()
    response = view.json_error_response(error_message=error_input, status=400)
    assert response.status_code == 400
    assert json.loads(response.content) == expected


class RetrieveTestView(ModelRetrieveMixin):
    model_class = Workflow
    serializer_method = lambda self, x: {"id": str(x.id)}


@pytest.mark.django_db
def test_get_object_found():
    obj = baker.make(
        Workflow,
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
    )
    view = RetrieveTestView()
    result = view.get_object(obj.pk)
    assert result.id == obj.id


def test_get_object_not_implemented():
    view = ModelRetrieveMixin()
    with pytest.raises(NotImplementedError):
        view.get_object("123")


class ListTestView(ModelListMixin):
    model_class = Workflow
    serializer_method = lambda self, x: {"id": str(x.id)}


@pytest.mark.django_db
def test_get_queryset_returns_all():
    baker.make(
        Workflow,
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
        _quantity=3,
    )
    view = ListTestView()
    request = HttpRequest()
    qs = view.get_queryset(request)
    assert qs.count() == 3


class CreateTestView(ModelCreateMixin):
    model_class = Workflow
    serializer_method = lambda self, x: {"id": str(x.id)}
    required_fields = ["name", "workflow_definition"]


@pytest.mark.django_db
def test_create_object_success():
    data = {
        "name": "ChainBuilder",
        "workflow_definition": [{"type": "prompt", "name": "Test"}],
    }
    view = CreateTestView()
    obj = view.create_object(data)
    assert obj.pk is not None


def test_create_object_missing_required_field():
    view = CreateTestView()
    with pytest.raises(ValidationError):
        view.create_object({"name": "Missing"})


@pytest.mark.django_db
def test_update_object_success():
    obj = baker.make(
        Workflow,
        name="OldName",
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
    )
    view = ModelUpdateMixin()
    view.update_object(obj, {"name": "NewName"})
    obj.refresh_from_db()
    assert obj.name == "NewName"


@pytest.mark.django_db
def test_delete_object_success():
    obj = baker.make(
        Workflow,
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
    )
    view = ModelDeleteMixin()
    view.delete_object(obj)
    assert not Workflow.objects.filter(pk=obj.pk).exists()


class ActivationView(ModelActivateDeactivateMixin, RetrieveTestView, JSONResponseMixin):
    model_class = Workflow
    serializer_method = lambda self, x: {"id": str(x.id), "is_active": x.is_active}


@pytest.mark.django_db
def test_activation_success():
    obj = baker.make(
        Workflow,
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
        is_active=False,
    )
    view = ActivationView()
    request = RequestFactory().post("/activate/")
    response = view.post(request, obj.pk, "activate")
    assert response.status_code == 200
    assert Workflow.objects.get(pk=obj.pk).is_active is True


@pytest.mark.django_db
def test_invalid_action():
    obj = baker.make(
        Workflow,
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
    )
    view = ActivationView()
    request = RequestFactory().post("/invalid/")
    response = view.post(request, obj.pk, "invalid")
    assert response.status_code == 400


@pytest.mark.django_db
def test_object_not_found():
    view = ActivationView()
    request = RequestFactory().post("/activate/")
    response = view.post(request, "nonexistent", "activate")
    assert response.status_code == 404
