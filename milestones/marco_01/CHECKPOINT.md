# Marco 01 — continuação aprovada com duas emendas

**Estado congelado em 09/09/2026. Tag: `v0.1.0-marco-01`.**

O usuário assistiu e aprovou `chain_second_join_joined_00001_.mp4`. A cópia exata está em `artifacts/03_porta.mp4`, com SHA-256 em `manifest.json`. São **430 frames, 24 fps, aproximadamente 17,917 segundos**.

Este marco é utilizável. Experimentos de memória semântica ou contextos maiores não fazem parte desta versão.

## Receita preservada

| Etapa | Ação | Seed | Target | Contexto | Filme acumulado |
|---|---|---:|---:|---:|---:|
| 01 | Caminhar ao lado da parede | 12345 | 124 frames | 0 | 124 frames |
| 02 | Parar, abrir a bolsa, retirar chave, arco e aproximação da câmera | 34567 | 175 frames | 22 | 277 frames |
| 03 | Ir até a porta, usar a chave e entrar, câmera acompanha | 56789 | 175 frames | 22 | 430 frames |

Todas: FL2VA INT8 ConvRot, Turbo FL2V 8step v1.0, força 1, **8 steps**, Euler/simple, 736×416. Etapas 02/03: `pinned_av`, áudio de contexto 1 segundo. Prompts completos e todas as ligações estão nos JSON.

## Reproduzir em outra máquina

1. Instale o pack em `ComfyUI/custom_nodes/h3_continuity` e reinicie o ComfyUI. A versão validada foi ComfyUI 0.34.0, commit registrado no manifesto. Utilize uma instalação compatível com H3 nativo e suas máscaras AV.
2. Disponibilize os modelos listados em `manifest.json`. Os pesos não estão incluídos no repositório. Não troque para a LoRA Turbo 768p nem para a de referência se quiser manter esta receita.
3. No terminal, execute usando o Python da sua instalação:

   ```bash
   python ComfyUI/custom_nodes/h3_continuity/scripts/restore_checkpoint.py --comfy-root ComfyUI
   ```

   Substitua os caminhos pelos seus. Windows portable: use `python_embeded/python.exe`. O script copia somente os checkpoints e os vídeos de entrada deste marco, verifica seus hashes e não substitui arquivos diferentes.
4. Abra **`03_porta.json`** no ComfyUI e execute. Ele parte do latent e do vídeo exatos da etapa 02. Os novos resultados ficam em `output/h3_continuity_reproduced/`; o checkpoint aprovado fica preservado.

Para repetir as etapas anteriores, há `01_fonte.json` e `02_bolsa.json`. Os arquivos `.api.json` são os equivalentes para `/prompt`, não os arquivos do canvas. Por padrão, cada etapa de continuação usa o checkpoint aprovado anterior. Para encadear seus novos takes, altere o Load Latent e o vídeo de entrada para os novos arquivos.

Mesmo com seed igual, outras versões, kernels, GPU, precisão e codecs podem produzir diferenças numéricas. **A reprodução exata do resultado já aprovado é a cópia dos artefatos com os hashes registrados.** Os workflows preservam a receita para regenerar, não prometem equivalência binária entre hardwares.

## Continuar depois da porta

Após restaurar:

- `Load Latent`: `h3_continuity_checkpoints/marco_01/03_porta.safetensors`.
- `Load Video`: `h3_m01_porta.mp4`.
- Mantenha Prepare em `pinned_av`, contexto 22, áudio 1 s e a mesma resolução.
- Escreva o próximo movimento. Ligue o **plan** ao Save Latent para carregar adiante a posição na linha do tempo.

Use `03_porta.json` como base e troque esses dois campos e o prompt. O latent salvo do sampler permanece inteiro; o Assemble recorta apenas as imagens e áudio destinados à montagem.

## Achados deste marco

- A cauda correta importa: o Add Guide nativo reduz clipes à grade de H3 usando o começo do lote, o que pode excluir o verdadeiro final.
- Preservar o prefixo por máscara mostrou boa continuidade sem impedir uma nova ação ou movimento de câmera.
- Transportar o latent evita uma recodificação VAE em cada etapa.
- `pinned_av` preservou melhor o áudio de contexto do que guias isolados nos testes realizados, mas o áudio de Turbo continua sujeito à qualidade do modelo.
- Mais contexto não foi automaticamente melhor: 22 frames foi o melhor ponto de partida medido neste conjunto.
- Uma gravação original de 48 kHz foi preservada amostra por amostra na montagem FLAC. Isso não demonstra lip sync perfeito.

Pesquisa e validação detalhadas: `../../docs/PESQUISA.md` e `../../docs/TESTES.md`. Foram aprovados 40 testes de código. Não há garantia de ausência total de artefatos em outras cenas.

## Rotina para próximos marcos

Preservar código em commit e tag; prompts, workflows, seeds e modelos em manifesto; latents brutos e vídeo montado; hashes dos artefatos; métricas e observações em um novo `CHECKPOINT.md`. Só marcar como novo marco o resultado efetivamente avaliado. O Marco 01 permanece intocado como referência.
