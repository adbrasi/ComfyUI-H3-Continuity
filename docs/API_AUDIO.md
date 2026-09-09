# Integração de áudio na API — H3 Continuity

Código na branch main; mudança funcional introduzida em 27cb880, tag research-audio-masks-01. Atualize o pack e reinicie ComfyUI quando não houver execução ativa. Consulte `/object_info/H3ContinuityPrepare` para verificar o campo `missing_audio`. Nenhum pacote novo é necessário.

## Campos do Prepare

Os IDs dos nodes e os índices de saída existentes permanecem iguais. Os novos widgets são opcionais e foram acrescentados ao final.

| Campo | Tipo / valores | Significado |
|---|---|---|
| soundtrack | AUDIO | Gravação fornecida pelo usuário, via saída de LoadAudio. |
| soundtrack_mode | full_video (padrão), continuation_only | full_video começa no tempo zero da linha do tempo; continuation_only começa na emenda desta chamada. |
| missing_audio | generate (novo padrão), silence | generate libera os intervalos sem gravação dentro da janela do H3; silence conserva o comportamento anterior. |
| source_audio | AUDIO opcional | Som do vídeo que será continuado. Não é a trilha do resultado inteiro. |
| audio_vae | VAE | VAE de áudio MiniMax H3; necessário com soundtrack. |
| source_end_seconds | FLOAT, padrão 0 | 0 infere o final da fonte/metadados. Em cadeias ou recortes, representa a posição absoluta do final da fonte na linha do tempo. |
| feather_frames | INT, padrão 0 | Feather de vídeo, independente deste ajuste de áudio. Manter 0 para o método aprovado. |

Sem soundtrack, o campo missing_audio não ativa a montagem de trilha parcial; permanece o fluxo normal de áudio da continuação. Em full_video, source_audio não substitui o arquivo escolhido para a trilha completa. Em continuation_only, source_audio preserva o som anterior; o plan transporta o source_audio ligado ao Prepare caso ele não esteja conectado diretamente ao Assemble.

Não substitua um áudio ausente por um WAV silencioso. Omita source_audio ou use H3ContinuityImport: ele distingue faixa ausente de silêncio real por metadados internos de duração válida. Pausas e zeros dentro de uma gravação fornecida são preservados.

## Ligações obrigatórias para aproveitar o áudio gerado

- Prepare saída0 (positive) → guider.conditioning.
- Prepare saída1 (latent) → sampler.latent_image.
- Prepare saída2 (plan) → Assemble.plan. O plano é um objeto interno do grafo; não o construa ou serialize manualmente na API.
- Sampler → VAEDecode e VAEDecodeAudio, com seus respectivos VAEs.
- VAEDecode saída0 → Assemble.images.
- **VAEDecodeAudio saída0 → Assemble.audio.** É o decode completo, incluindo contexto. Sem ele, os intervalos sem gravação não podem ser preenchidos na montagem.
- Fonte de vídeo normalizada → Assemble.source_images para entregar origem + continuação. Sem source_images, entrega somente a continuação.
- Assemble saída0 → CreateVideo.images; Assemble saída1 → CreateVideo.audio; fps24.

Não conecte a gravação do usuário diretamente a Assemble.audio. Essa entrada tem o tempo do sampler e inclui a sobreposição. Use soundtrack no Prepare; o plan leva as amostras exatas ao Assemble.

## Exemplos API completos

- [full_video](../workflows/03_audio_full_video.api.json)
- [continuation_only](../workflows/04_audio_continuation_only.api.json)

São grafos no formato prompt do ComfyUI; envie `{"prompt": grafo, "client_id": "seu-cliente"}` para POST /prompt. Os exemplos têm nomes de modelos e arquivos desta máquina: adapte esses caminhos ao ambiente de destino. O WAV de tons é uma demonstração; substitua pelo arquivo do usuário já disponível em input/.

Nos exemplos, altere `31.inputs.audio` para o nome do arquivo e `21.inputs.soundtrack_mode` / `21.inputs.missing_audio` para a política escolhida. Os nodes21 e24 já têm as ligações acima; `25.inputs.audio` aponta para a saída1 do Assemble.

## Semântica e duração

O arquivo fornecido condiciona a geração: o Prepare codifica seu áudio e aplica máscara0 aos tokens cobertos, máscara1 aos trechos sem gravação. Tokens de borda têm cobertura proporcional na grade de 40 Hz (25 ms). O Assemble usa as amostras fornecidas nos intervalos cobertos e o decode gerado nos demais. Não se usa amplitude para decidir se há áudio.

Áudio maior que a entrega é cortado. Áudio menor deixa o restante livre para gerar, quando missing_audio=generate. Para manter zeros no restante, envie explicitamente missing_audio=silence. Se as taxas/canais forem diferentes, a montagem compatibiliza-os; a codificação final do vídeo também pode comprimir o áudio.

Exemplo: fonte3 s, contexto22 frames, alvo124 frames. A entrega tem 3 + (124-22)/24 = 7,25 s. A janela do modelo começa em 3-22/24 = 2,0833 s e termina em 7,25 s.

- full_video com arquivo4 s: preserva [0,4) s e usa geração entre4 e7,25 s.
- continuation_only com arquivo0,5 s e sem source_audio: gera áudio no contexto entre2,0833 e3 s; preserva a gravação entre3 e3,5 s; usa geração entre3,5 e7,25 s.
- No segundo exemplo, os primeiros2,0833 s não participam da geração e continuam sem som. O report do Assemble informa o trecho descoberto. Para gerar som para essa parte, ela precisa participar de outra janela/geração que a cubra. A API não deve prometer que o modelo sonoriza frames fora da sua janela.

O length do node nativo é o tamanho total da janela, incluindo contexto; não é apenas a quantidade de frames novos. O contexto efetivo é ajustado à grade H3 (5+17k), então use os valores efetivos produzidos pelo fluxo, e não apenas o valor solicitado. Em full_video, mantenha os metadados de tempo ao encadear; em continuation_only, cada gravação nova começa na emenda da chamada atual.

## Validação e compatibilidade

76 testes passaram; três renders FL2VA + Turbo, 8 steps, Euler/simple, 736×416. Nos FLACs, a montagem coincidiu exatamente com a gravação e o decode gerado nos intervalos esperados. Essa validação não promete sincronização labial perfeita ou qualidade do áudio Turbo.

[Checkpoint, inputs e resultados](../milestones/research_audio_masks_01/CHECKPOINT.md). A ausência do campo missing_audio agora significa generate: clientes que dependiam de completar silêncio precisam enviar silence explicitamente. Os checkpoints e tags anteriores permanecem intactos.
