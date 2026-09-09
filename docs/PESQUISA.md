# Pesquisa: continuação MiniMax H3

Inspeção de código e documentação em 8–9 de setembro de 2026. As conclusões abaixo distinguem mecanismos implementados de promessas perceptuais dos autores. Os benchmarks deles não foram tratados como resultados nossos.

## Add Guide nativo

O `MiniMaxH3AddGuide` instalado já aceita imagens, clipes e áudio como keyframes. Esses dados condicionam a geração em posições da linha do tempo. Isso é diferente de uma máscara zero sobre os próprios tokens do target: os tokens gerados continuam podendo diferir dos guides.

O node alinha clipes para baixo à grade `5 + 17k` e codifica `image[:guide_frames]`. Em 72 frames, guarda os primeiros 56 e descarta os últimos 16. A 24 fps, são 0,667 segundos de movimento final ausentes. Não é um defeito para todos os usos do Add Guide, mas exige que quem quer continuar um vídeo selecione a cauda antes. O áudio começa em `frame_idx`, também não é automaticamente uma janela terminando na emenda. O node não monta as duas partes nem elimina a repetição.

Fonte exata instalada: `comfy_extras/nodes_minimax_h3.py`, classe `MiniMaxH3AddGuide`. O layout e o suporte a máscaras foram conferidos em `comfy/ldm/minimax/model.py` e `comfy/model_base.py`.

## Comparação dos projetos

| Projeto | Diferença principal em relação ao Add Guide | Adequação ao objetivo |
|---|---|---|
| [Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | Seleção da cauda, transporte direto dos latents AV, áudio posicionado na linha do tempo, trim e checkpoints | Base conceitual enxuta para emendas; keyframes não tornam os tokens do target imutáveis |
| [Context Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) | Gerenciamento de cenas/takes, revisão, retomada, armazenamento e prefixos AV com máscaras, incluindo variações de suavização | Mais completo para uma produção longa, mas também com superfície de código e interface maior |
| [Inpaint Tools](https://github.com/panghea/ComfyUI-MiniMax-H3-Inpaint-Tools) | Máscaras separadas para vídeo/áudio, regiões, intervalos e extensão do target | Muito relevante para preservar partes conhecidas. A documentação admite que os testes de extensão são limitados |
| [Multishot](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow) | Automação de vários shots, memória/contexto, controle de deriva e gravação original | CORE é relativamente simples; FULL acrescenta vários packs. Mais abrangente do que apenas continuar um clipe |
| [Hybrid Cond](https://github.com/kitsune123150/minimax-h3-hybrid-cond) | Combina referências e first/last-frame no mesmo payload | Útil para identidade e composição, não resolve sozinho a emenda temporal |
| [Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender) | Continuação com gestão de clipes, cache em disco, referências por clipe e decodificação final | Automatiza a produção. Interface simples pode esconder bastante infraestrutura |

### Instalação e compatibilidade

Não é correto dizer que todos exigem dezenas de dependências. Motion Context e o Multishot básico aproveitam bibliotecas do ComfyUI. O FULL do Multishot acrescenta packs para roteiro, scheduler e otimizações; são recursos extras. Inpaint Tools declara `minimax-h3-latent-core`; Extender declara `imageio-ffmpeg`. A complexidade de uso e o tamanho do código não equivalem ao número de pacotes Python necessários. [Multishot: requisitos](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow), [Inpaint Tools](https://github.com/panghea/ComfyUI-MiniMax-H3-Inpaint-Tools), [Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender).

O Hybrid Cond consultado instala um wrapper global em `MiniMaxH3.extra_conds`. Ele acessa `item['latent']` em todos os keyframes, o que conflita com keyframes somente de áudio. O ComfyUI desta máquina já concatena vídeo/áudio de keyframes e referências corretamente; esse patch não é necessário aqui e não foi instalado. Essa conclusão vem da inspeção dos dois códigos, não de um render com o Hybrid instalado. [Código do patch](https://github.com/kitsune123150/minimax-h3-hybrid-cond/blob/main/model_base_patch.py).

A versão atual de Motion Context evita patches globais. Context Loop inclui verificações e compatibilidade para suporte a máscaras, inclusive a correção de velocidade com máscaras fracionárias. Isso explica parte de seu tamanho; não é simplesmente código supérfluo. Nesta máquina, a correção já existe nativamente. [Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), [Context Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop).

## Decisão de implementação

H3 Continuity combina seleção da cauda, reaproveitamento de latents, posicionamento de áudio, máscara nativa do target e montagem sem repetir contexto. Mantém os loaders, condicionadores, sampler e VAEs normais. Não instala nenhum dos projetos pesquisados, não troca versões de CUDA/Torch, não usa serviços externos e não faz monkey-patching.

`pinned_prefix` preserva o vídeo conhecido e orienta o áudio. `pinned_av` também fixa as linhas inteiras de áudio que antecedem a emenda. `anchors` permite a comparação com o prefixo regenerado. Gravações originais usam máscara no áudio e as próprias amostras na montagem, pois um áudio parecido não equivale ao áudio original.

As máscaras não tornam a continuação futura infalível. O modelo ainda decide o que ocorre depois do prefixo. A duração do contexto, a compatibilidade do prompt com o estado anterior, a resolução e o conteúdo influenciam o resultado. O teste de ação nova e câmera serve para avaliar justamente esse limite.
