from typing import Mapping

from dify_plugin.entities.model import (
    AIModelEntity,
    I18nObject
)

from dify_plugin.interfaces.model.openai_compatible.text_embedding import (
    OAICompatEmbeddingModel,
)


class OpenAITextEmbeddingModel(OAICompatEmbeddingModel):

    def get_customizable_model_schema(self, model: str, credentials: Mapping) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label= I18nObject(
                en_US=credentials["display_name"],
                zh_Hans=credentials["display_name"]
            )

        return entity

    # remove user parameter for embedding models
    def _prepare_embedding_payload(self, **kwargs):
        """
        Prepare embedding payload and remove user parameter for embedding models
        """
        payload = super()._prepare_embedding_payload(**kwargs)
        if 'user' in payload:
            del payload['user']
        return payload
