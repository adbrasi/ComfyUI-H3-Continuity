# Pesquisa 02 — nova entidade, interação e câmera

Continuação do checkpoint `research-memory-01`, sem mudar o código do node. O usuário propôs pedir acontecimentos diferentes dos da fonte para separar memória visual de capacidade de executar uma ação nova.

Fonte do teste anterior: jardim, mulher sentada no banco, cachorro azul alado que aparece no início e sai. Prompt novo: uma pessoa de casaco laranja entra empurrando uma bicicleta vermelha; o cachorro volta, aproxima-se da roda e senta; a mulher levanta e cumprimenta; a câmera sobe do chão ao nível dos olhos e faz um arco de aproximadamente 45 graus. As cores das asas e do acessório do cachorro não são dadas no prompt.

Configuração: FL2VA first/last-frame + Turbo FL2V, 8 steps, Euler/simple, 736×416, 24 fps, seed 90123. Todos os métodos entregam 204 frames novos (8,5 s), montados após 90 frames da fonte, total 294 frames (12,25 s). Prefixo 22 usa target 226; prefixo/guide90 usa target 294. Não há condicionamento de áudio anterior.

## Resultado e avaliação

| Método | Tempo completo observado |
|---|---:|
| Prefixo22 |53,05 s|
| Prefixo90 |74,98 s|
| Prefixo22 + imagem no encoder |53,21 s|
| Add Guide90 |108,81 s|

Mesma duração nova, resolução e seed. Tempos de execução com o estado de cache da sessão; não são benchmark aleatorizado.

- `tail22`: pessoa, bicicleta e mudança de câmera aparecem, mas o cachorro volta preto e branco, sem a aparência original.
- `tail90`: cachorro azul/alado e nova ação aparecem, porém surge uma pessoa adicional de casaco laranja. Preservar aparência não basta para considerar o resultado correto.
- `encoder_image`: 22 frames finais congelados mais uma imagem antiga (frame 8) enviada **somente ao encoder visual nativo**, sem VAE de referência. Recupera a aparência aproximada do cachorro, introduz pessoa/bicicleta, eleva e desloca a câmera e mantém as duas pessoas esperadas nos frames inspecionados.

- `guide90`: recupera o cachorro e introduz pessoa/bicicleta, mas a montagem salta do jardim vazio para um enquadramento próximo da bicicleta já em cena no primeiro frame novo. A aparência recuperada não compensou essa falha de emenda nesta seed.

O usuário assistiu a `memory_action_encoder_image_90123_joined_00001_.mp4` e considerou esse o melhor resultado até então. A inspeção por folha de contato também o favorece entre os três acima. Não medimos o arco em graus nem comprovamos contato físico perfeito com a roda. Ainda é uma única seed neste teste difícil, sem garantia geral de ausência de flicker, deriva ou entidades extras.

O condicionamento visual do encoder é uma capacidade nativa do H3/ComfyUI. O pack preserva o prefixo, mantém o conditioning de referência e cuida da montagem/checkpoint. Não há implementação de FreeMem, FramePack ou outro paper neste resultado. O frame de memória foi escolhido manualmente; recuperação automática por embeddings ainda é uma proposta.

## Reproduzir ou continuar o resultado aprovado

1. Instale a branch de pesquisa e os pesos documentados no Marco 01; reinicie ComfyUI.
2. Restaure a fonte exata com `python milestones/research_memory_01/restore_inputs.py --comfy-root /caminho/ComfyUI`.
3. Execute o workflow UI em `ui/` ou envie `api/memory_action_encoder_image_90123.api.json` ao `/prompt` usando `milestones/research_memory_01/submit_api.py`.
4. O vídeo é salvo em `output/h3_continuity_tests/`, acessível nesta máquina por `/workspace/H3-testes/`.

O latent bruto do resultado aprovado também está em `artifacts/`, com os dois streams e metadados de tempo. Para usá-lo como origem de outra continuação, copie para um caminho dentro de `ComfyUI/output/` e abra com H3 Continuity Load Latent. Ele inclui 22 frames de contexto interno: não concatene o vídeo decodificado inteiro dele a uma montagem anterior. Use a montagem aprovada como source_images/source_audio de Assemble ou salve somente a nova extensão.

Workflows API executados; workflow UI conferido estruturalmente, sem operação manual em outra máquina. Tempos observados em `results.json`; arquivos e hashes em `manifest.json`.
