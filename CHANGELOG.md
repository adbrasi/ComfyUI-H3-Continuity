# 0.2.0

- Montagem entrega os frames de contexto liberados pelo feather, corrigindo o descarte do reparo.
- Bridge com contexto latente nas duas bordas e montagem sem duplicar overlap.
- Retake aceita MASK espacial estática ou por frame e força visual independente do áudio.
- Assemble Latents acumula o filme AV sem recodificação e remove máscaras antigas de sampling.
- Checkpoints de conditioning/referências em safetensors com metadados JSON, sem pickle.
- Joint Refine experimental funde predições de janelas antes de cada avanço do sampler.
- Workflows de bridge, retake, máscara espacial, refino e âncora futura; scripts de reprodução.
- Validação atual documentada em `milestones/gpu_validation/RESULTADOS.md`.

# 0.1.0

- Continuação pela cauda de imagens ou latents H3 AV.
- Máscaras nativas de vídeo e áudio, e alternativa com anchors.
- Importação a 24 fps, montagem sem contexto duplicado e checkpoints safetensors.
- Gravação original fixada no target e preservada na montagem.
- Dois workflows ComfyUI, 40 testes e demonstrações aprovadas com Turbo/8 steps.
