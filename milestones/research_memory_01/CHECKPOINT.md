# Pesquisa 01 — contexto maior e memória visual

2026-09-09. Branch `research/memory-context`, baseada em `v0.1.0-marco-01`. Este é um checkpoint de pesquisa. O Marco 01 aprovado permanece intacto. Não há treinamento, patch de atenção, cache KV, interpolação ou instalação de outro modelo.

## O que foi implementado

`context_frames` deixou de ser uma lista restrita a 5/22/39/56 e passou a ser um inteiro. O node informa quantos frames realmente usou quando precisa ajustar o pedido. A seleção continua sendo da cauda verdadeira, alinhada à grade temporal `5 + 17k`.

112 pedidos → últimos 107 utilizados. A alternativa maior válida é 124. A interface admite até 3600, acompanhando o limite de entrada nativo; isso **não significa** qualidade validada nesse tamanho. Renderizamos 90 e 107 frames de contexto. Testes de estrutura também cobrem 124, 243 e 362; esses testes não são renders. O contexto precisa caber no target e deixar espaço para frames novos. A interface nativa indica aproximadamente 124–362 frames como faixa de duração de treinamento, não como garantia de continuação.

Contagem: `novos = target - contexto_usado`; `montagem = fonte_completa + novos`. A imagem de memória separada não ocupa frames da montagem. Um pedido aproximado de 10 segundos no target é alinhado para 243 frames (10,125 s). Se o contexto efetivo for 107, restam 136 frames novos (5,667 s). Para obter cerca de 10 segundos novos, é necessário somar o contexto ao target e considerar a grade.

## Desenho do experimento do cachorro

Fonte: 90 frames / 3,75 s, 736×416, 24 fps. Um cachorro azul de corpo baixo, orelhas caídas, asas brancas e acessório amarelo com detalhe vermelho aparece no início e sai. Ele está ausente dos últimos 22 frames. A continuação pede apenas que o cachorro azul alado volte e interaja com a mulher. Raça, cores das asas, acessório e detalhe vermelho não estão no prompt de continuação.

Pesos first/last-frame FL2VA INT8 ConvRot; Turbo FL2V 8step v1.0, 8 steps, Euler/simple. Mesmo prompt em todas as continuações. Seeds 89123 e 89124. Áudio de contexto desligado; áudio novo do Turbo não é critério de qualidade aqui. Os nomes/hashes dos modelos estão no manifesto do Marco 01.

- `tail22`: últimos 22 frames como prefixo fixo; target 124.
- `tail90`: todos os 90 frames como prefixo fixo; target 192.
- `encoder_image`: prefixo 22 + frame 8 da fonte no encoder visual Qwen nativo, sem VAE na referência; target 124.
- `encoder_latent_image`: igual, com VAE de referência conectado; imagem 736×416.
- `encoder_latent_small`: igual, com imagem de referência reduzida para 352×192 antes do encoder/VAE; prefixo e geração continuam 736×416.
- `guide90`: Add Guide nativo com os 90 frames decodificados, `frame_idx=0`, target 192; nenhum latent/máscara de Prepare chega ao sampler. Prepare calcula somente o plano da montagem.
- `anchors90`: mesmo histórico 90, reaproveitado como guide latent; target 192.
- `pinned_pixels90`: mesmo histórico 90 decodificado e recodificado por VAE, depois fixado por máscara; target 192. Este controle separa o efeito de evitar recodificação do efeito da máscara.

Todas as variantes acrescentam **102 frames / 4,25 s**, produzindo 192 frames / 8 s. O primeiro frame novo é o 90. A geração curta e a longa têm formatos de ruído diferentes; mesma seed não implica ruído idêntico elemento a elemento entre durações diferentes. Guide90 e os prefixos 90 têm o mesmo formato de target.

## Resultados observados

Nas duas seeds, tail22 retornou um cachorro preto e branco de pelo comprido, sem a aparência original. Tail90 e as três variantes de imagem antiga recuperaram cachorro azul, asas brancas e aparência aproximada do acessório/corpo. O usuário assistiu e reconheceu o cachorro nos dois tail90 e no encoder_image 89124, entre outros.

Isso é evidência de condicionamento útil, não demonstração de uma memória nova inventada pelo pack. **Add Guide com 90 frames também recuperou a aparência nas duas seeds.** Os tempos completos e as observações estão em RESULTADOS.md.

Controle com referência trocada: geramos outra fonte com cachorro azul de pernas altas/orelhas eretas, asas pretas e acessório roxo. Mantendo a fonte original como prefixo, o prompt e seed 89123, trocar apenas a imagem antiga mudou a aparência na direção da referência trocada, tanto no encoder sozinho quanto no encoder+latent pequeno. O fundo da imagem alternativa também difere: não é uma edição perfeitamente isolada do sujeito. Ainda assim, a transferência dos atributos não descritos reforça a interpretação de influência visual, em vez de mera coincidência do prompt.

A avaliação visual usa folhas de contato, além do julgamento dos vídeos pelo usuário. Não declaramos identidade exata de todos os detalhes — o pequeno pingente frequentemente fica oculto — nem ausência de flicker. Algumas emendas mudam o enquadramento. `seam_ratio` mede diferença de pixels e fica inflado em cenas quase paradas; não deve ordenar a qualidade sozinho. O erro de latent mede conservação do prefixo, não qualidade do futuro.

## O que difere do Add Guide

Add Guide acrescenta tokens de condicionamento de vídeo e permite que o target inteiro seja gerado. Nosso modo anchors usa o mesmo mecanismo de guias, com transporte direto do latent e preparação da cauda. O modo pinned copia o passado para o próprio target e usa a máscara nativa para preservá-lo: evita duplicar esse passado como um bloco extra de guides e evita recodificar quando há um checkpoint latent. Os tokens fixos continuam participando da atenção e custando processamento.

O pack também escolhe a cauda verdadeira, alinha o áudio, remove a repetição de contexto na montagem e salva ambos os streams com a posição temporal. São decisões práticas de engenharia, não capacidades aprendidas novas. O encoder de imagem e as referências latent usados aqui já existem no ComfyUI; o checkpoint de difusão continua sendo FL2VA, apesar do nome do condicionador Reference to Video.

## Próxima hipótese

Contexto recente preserva o movimento; poucas imagens antigas podem recuperar aparência com menos custo que reenviar toda a história. Selecionar essas imagens por relevância, evitar redundância, manter limites de tokens e permitir que o usuário fixe uma referência são próximos recursos possíveis. Aqui o frame 8 foi escolhido manualmente; não existe seleção semântica automática implementada.

Embeddings de outros modelos podem ajudar a buscar frames. Não devem ser concatenados arbitrariamente ao conditioning do H3: representações com o mesmo tamanho podem ter significados incompatíveis. O caminho visual nativo já testado é a opção mais simples. O relatório de oito papers está em `../../docs/PAPERS_MEMORIA.md`; FreeMem/Ouroboros inspiram memória de features, mas exigiriam adaptações e validação próprias no H3 Turbo. FramePack completo exige adaptação/treino, não é uma troca de node training-free.

## Reproduzir

1. Instale o pack desta branch em `ComfyUI/custom_nodes/h3_continuity` e reinicie ComfyUI. Use a instalação/versões do Marco 01 como base.
2. Execute `python milestones/research_memory_01/restore_inputs.py --comfy-root /caminho/ComfyUI`. Ele verifica SHA256 e restaura as três fontes exatas, sem sobrescrever arquivos diferentes.
3. Abra um workflow de `ui/` no ComfyUI. Eles usam os mesmos nodes e parâmetros dos testes. Alternativamente execute `python milestones/research_memory_01/submit_api.py milestones/research_memory_01/api/memory_tail90_89123.api.json --url http://127.0.0.1:8188`.
4. Os vídeos são gravados em `ComfyUI/output/h3_continuity_tests/`. Na máquina desta pesquisa, esse diretório também é `/workspace/H3-testes`.

Os workflows API foram executados. Os workflows UI foram gerados a partir deles e conferidos estruturalmente; ainda não foram operados manualmente em outra máquina. Nomes de modelos precisam coincidir com os instalados. Mesmo com pesos, seed e versões iguais, kernels/hardware diferentes podem introduzir pequenas diferenças.
