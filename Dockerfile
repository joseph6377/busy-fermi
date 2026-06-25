FROM runpod/worker-comfyui:5.8.6-base

COPY docker/link-cached-models.sh /link-cached-models.sh
RUN chmod 0755 /link-cached-models.sh

CMD ["/link-cached-models.sh"]
