# ComfyUI H3 Continuity

Continuação, bridges e retakes para MiniMax H3 usando latentes e máscaras nativas do ComfyUI. Training-free: o pack não treina pesos, não exige LoRAs adicionais e não altera globalmente o modelo ou o sampler.

O objetivo é gerar o movimento da transição e entregar os frames que o modelo reparou. Preservar latentes evita a recodificação VAE em cada extensão; referências canônicas e bridges para âncoras futuras compatíveis ajudam a controlar a deriva. **Não há garantia de continuidade ilimitada sem degradação.**

## Instalação

Com o ComfyUI atualizado e o H3 nativo funcionando, execute dentro de `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/adbrasi/ComfyUI-H3-Continuity.git
```

Reinicie o ComfyUI e procure **H3 Continuity**, **H3 Bridge**, **H3 Retake** e **H3 Joint Refine**. As dependências de execução já pertencem ao ComfyUI. É necessário suporte nativo a latentes H3 AV e máscaras por token; não basta um workflow de API remota MiniMax.

## Começar

- [01_continuar_video.json](workflows/01_continuar_video.json): continuar um vídeo externo. Import converte para 24 fps; o contexto passa uma vez pelo VAE.
- [02_continuar_latent.json](workflows/02_continuar_latent.json): continuar um checkpoint AV sem recodificar frames. Use a saída completa do sampler.
- [03_audio_full_video.json](workflows/03_audio_full_video.json) e [04_audio_continuation_only.json](workflows/04_audio_continuation_only.json): orientar a geração com uma gravação e conservar suas amostras na montagem.
- [05_bridge.json](workflows/05_bridge.json) e [09_future_anchor.json](workflows/09_future_anchor.json): ponte entre clipes e geração de uma âncora futura independente.
- [06_retake.json](workflows/06_retake.json) e [07_retake_spatial.json](workflows/07_retake_spatial.json): retake temporal e espacial.
- [11_retake_differential.json](workflows/11_retake_differential.json) e [12_retake_spatial_differential.json](workflows/12_retake_spatial_differential.json): os mesmos retakes com Differential Diffusion experimental.
- [10_assemble_latents.json](workflows/10_assemble_latents.json): conservar a sequência completa como um canvas AV para retake e refino posteriores.
- [08_joint_refine.json](workflows/08_joint_refine.json): refino por janelas de um canvas AV completo.

Os exemplos contêm nomes de modelos e caminhos da máquina de origem: selecione seus arquivos ao abrir. Os arquivos `.api.json` são grafos para a API; os `.json` acima são workflows do canvas.

| Operação | Nodes |
|---|---|
| Continuação: pinned_av, pinned_prefix ou anchors | Seletor `method` no Prepare → sampler nativo → Assemble |
| Ponte entre dois clipes | Bridge Prepare → sampler → Bridge Assemble |
| Refazer uma região | Retake Prepare Mask → sampler → Retake Assemble |
| Retake espacial | Entrada MASK do Retake; estática ou uma máscara por frame |
| Retomar trabalho | Save/Load Latent e Save/Load Conditioning |
| Acumular o filme em latent | Assemble Latents → Save Latent; resultado usado na próxima etapa |
| Refino conjunto experimental | Joint Refine → MODEL do sampler; mesma janela AV completa em LATENT |

Leia o [guia de continuação e retake](docs/continuation-retake.md) para conexões, grades temporais, handover do feather, bridges e limites do refino conjunto. O [guia de áudio](docs/AUDIO_ORIGINAL.md) detalha a montagem da trilha original.

Para refazer ou refinar uma cadeia inteira, acumule suas etapas com **Assemble Latents**. Um checkpoint da última janela isolada contém apenas essa janela; não recupera automaticamente os clipes anteriores.

O [marco anterior](milestones/marco_01/CHECKPOINT.md) e os [experimentos anteriores de retake](milestones/research_retake_01/CHECKPOINT.md) permanecem como registros históricos. Seus resultados não validam automaticamente as mudanças atuais.

## Métodos validados

Todos os métodos dos comparativos estão disponíveis neste pack. No workflow 02, altere `method` no **H3 Continuity · Prepare** entre `pinned_av`, `pinned_prefix` e `anchors`. Os exemplos 01/02 abrem com feather 17 e força 1; `anchors` não aplica feather. Bridge tem seu próprio Prepare/Assemble. Nos retakes, conecte MASK para limitar a edição espacial; os workflows 11/12 mostram o adaptador Differential Diffusion e seus SIGMAS. Nenhum método foi removido ou declarado vencedor.

Os [testes de extensão](milestones/method_validation/RESULTADOS.md) e [retake com portal](milestones/retake_validation/RESULTADOS.md) registram parâmetros, execução real em GPU e limites observados. Os MP4s e checkpoints dessas experiências ficam na máquina de teste; não acompanham o clone.
