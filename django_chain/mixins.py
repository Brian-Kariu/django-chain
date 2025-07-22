"""
Mixins for reusable view logic in django-chain.

This module provides mixins for JSON response handling, model CRUD operations, and activation/deactivation
patterns to be used in Django class-based views throughout django-chain.

Typical usage example:
    class MyView(JSONResponseMixin, ModelListMixin, View):
        ...
"""

import json
from typing import Any

from django.core.exceptions import ObjectDoesNotExist
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from django_chain.models import Prompt
from django_chain.models import Workflow


class JSONResponseMixin:
    """
    Mixin to render a response as JSON.
    Handles serialization of data and consistent error responses.
    """

    def render_json_response(
        self, data: dict = None, status: int = 200, safe: bool = True
    ) -> JsonResponse:
        """
        Renders data as a JSON response.
        `safe=False` is needed if the top-level object is a list.

        Args:
            data (dict, optional): Data to serialize. Defaults to None.
            status (int, optional): HTTP status code. Defaults to 200.
            safe (bool, optional): Whether to allow non-dict top-level objects. Defaults to True.

        Returns:
            JsonResponse: The JSON response.
        """
        return JsonResponse(data or {}, status=status, safe=safe)

    def json_error_response(self, error_message: str | dict, status: int = 400) -> JsonResponse:
        """
        Returns a consistent JSON error response.

        Args:
            error_message (str | dict): The error message or dict of errors.
            status (int, optional): HTTP status code. Defaults to 400.

        Returns:
            JsonResponse: The JSON error response.
        """
        if isinstance(error_message, dict):
            return JsonResponse({"errors": error_message}, status=status)
        return JsonResponse({"error": error_message}, status=status)

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        """
        Apply csrf_exempt to all methods in views inheriting this mixin.
        Also attempts to load JSON body for POST/PUT methods.

        Args:
            request (HttpRequest): The HTTP request object.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            HttpResponse: The response from the view.
        """
        if request.method in ["POST", "PUT"]:
            try:
                request.json_body = json.loads(request.body)
            except json.JSONDecodeError:
                return self.json_error_response("Invalid JSON in request body.", status=400)
        return super().dispatch(request, *args, **kwargs)


class ModelRetrieveMixin:
    """
    Mixin for retrieving a single model instance by its primary key (pk).
    Requires: `model_class`, `serializer_method`.
    """

    model_class = None
    serializer_method = None

    def get_object(self, pk: Any) -> Any:
        """
        Retrieve a model instance by primary key.

        Args:
            pk (Any): The primary key value.

        Returns:
            Any: The model instance or None if not found.
        """
        if self.model_class is None or self.serializer_method is None:
            raise NotImplementedError(
                "ModelRetrieveMixin requires 'model_class' and 'serializer_method' to be set."
            )

        try:
            obj = self.model_class.objects.get(id=pk)
            return obj
        except ObjectDoesNotExist:
            return None
        except ValidationError:
            return None


class ModelListMixin:
    """
    Mixin for listing model instances with optional filtering.
    Requires: `model_class`, `serializer_method`.
    """

    model_class = None
    serializer_method = None

    def get_queryset(self, request: Any) -> Any:
        """
        Get a queryset of all model instances.

        Args:
            request (HttpRequest): The HTTP request object.

        Returns:
            QuerySet: The queryset of model instances.
        """
        if self.model_class is None or self.serializer_method is None:
            raise NotImplementedError(
                "ModelListMixin requires 'model_class' and 'serializer_method' to be set."
            )

        return self.model_class.objects.all()

    def apply_list_filters(self, queryset: Any, request: Any) -> Any:
        """
        Apply filters to the queryset. Override in subclasses.

        Args:
            queryset (QuerySet): The queryset to filter.
            request (HttpRequest): The HTTP request object.

        Returns:
            QuerySet: The filtered queryset.
        """
        return queryset


class ModelCreateMixin:
    """
    Mixin for handling POST requests to create new model instances.
    Requires: `model_class`, `serializer_method`.
    """

    model_class = None
    serializer_method = None
    required_fields = []

    def create_object(self, request_data: dict) -> Any:
        """
        Create a new model instance from request data.

        Args:
            request_data (dict): Data for the new instance.

        Returns:
            Any: The created model instance.
        """
        if self.model_class is None or self.serializer_method is None:
            raise NotImplementedError(
                "ModelCreateMixin requires 'model_class' and 'serializer_method' to be set."
            )

        for field in self.required_fields:
            if field not in request_data:
                raise ValidationError(f"{field} is required.")

        obj = self.model_class(**request_data)
        obj.full_clean()
        obj.save()
        return obj


class ModelUpdateMixin:
    """
    Mixin for handling PUT requests to update existing model instances.
    Requires: `model_class`, `serializer_method`.
    """

    def update_object(self, obj: Any, request_data: dict) -> Any:
        """
        Update an existing model instance with request data.

        Args:
            obj (Any): The model instance to update.
            request_data (dict): Data to update the instance with.

        Returns:
            Any: The updated model instance.
        """
        for field, value in request_data.items():
            if hasattr(obj, field):
                setattr(obj, field, value)
        obj.full_clean()
        obj.save()
        return obj


class ModelDeleteMixin:
    """
    Mixin for handling DELETE requests to delete model instances.
    """

    def delete_object(self, obj: Any) -> None:
        """
        Delete a model instance.

        Args:
            obj (Any): The model instance to delete.
        """
        obj.delete()


class ModelActivateDeactivateMixin(JSONResponseMixin, ModelRetrieveMixin):
    """
    Mixin for handling activation/deactivation of models with an 'is_active' field.
    Requires: `model_class`, `serializer_method`.
    """

    model_class = None
    serializer_method = None

    def post(self, request: Any, pk: Any, action: str) -> Any:
        """
        Activate or deactivate a model instance.

        Args:
            request (HttpRequest): The HTTP request object.
            pk (Any): The primary key of the instance.
            action (str): 'activate' or 'deactivate'.

        Returns:
            JsonResponse: The response with the updated instance or error.
        """
        obj = self.get_object(pk)
        if obj is None:
            return self.json_error_response(f"{self.model_class} not found.", status=404)

        try:
            with transaction.atomic():
                if action == "activate":
                    obj.activate()
                elif action == "deactivate":
                    obj.deactivate()
                else:
                    return self.json_error_response(
                        "Invalid action. Must be 'activate' or 'deactivate'.", status=400
                    )
            return self.render_json_response(self.serializer_method(obj))
        except Exception as e:
            return self.json_error_response(str(e), status=500)
