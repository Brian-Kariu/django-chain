from django_chain.models import Prompt
from django_chain.models import Workflow
from django_chain.utils.llm_client import create_llm_chat_client


class LLMGenerationMixin:
    def get_model(self, **kwargs):
        prompt_info = kwargs.pop("prompt")
        self.prompt = Prompt.objects.filter(id=prompt_info).first()

        self.model_args = kwargs.pop("model")
        self.provider = self.model_args.pop("provider")
        self.model = create_llm_chat_client(self.provider, **self.model_args)
        # NOTE: We can add the vectorstore, embedding and other third party integrations here

        workflow_info = kwargs.pop("workflow")
        workflow = Workflow.objects.filter(id=workflow_info).first
        self.workflow_chain = workflow.get_chain()

        self.workflow_chain(self.prompt, self.provider, self.model_args)
        return self.workflow_chain

    def invoke_runnable(self, request, *args, **kwargs):
        runnable = kwargs["runnable"]
        user_input = kwargs["input"]
        output = runnable.invoke(user_input)
        return output

    def dispatch(self, request, *args, **kwargs):
        if request.method == "post" and "invoke" in request.PATH:
            kwargs["runnable"] = self.get_model(**kwargs)
            return super().dispatch(request, *args, **kwargs)
