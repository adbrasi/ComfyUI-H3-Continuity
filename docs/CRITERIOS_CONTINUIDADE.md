# Critérios de aceitação de continuação

Avaliação explícita do usuário em 2026-09-09, após o checkpoint `research-action-01`.

**Reprovado:** `memory_action_guide90_90123_joined_00001_.mp4`. O salto do jardim vazio para outro enquadramento, com a bicicleta já em cena, representa exatamente o comportamento que o usuário quer evitar. Recuperar a aparência do cachorro não compensa uma emenda descontínua.

**Referência positiva atual:** `memory_action_encoder_image_90123_joined_00001_.mp4`, considerado pelo usuário o melhor resultado até então. Usa 22 frames recentes fixos e uma imagem antiga somente no encoder visual nativo.

Para os próximos testes:

- A ação nova deve partir do estado espacial e temporal no fim da fonte. Entidades novas devem entrar de maneira visível e coerente.
- Mudanças de câmera são permitidas e desejáveis quando solicitadas, desde que percorram uma trajetória contínua a partir do enquadramento anterior.
- Salto de enquadramento ou avanço abrupto da ação na emenda reprova o resultado, mesmo quando identidade e aparência estão corretas.
- Avaliar também movimento, interação, entidades extras e deriva de aparência. Uma boa emenda sozinha não basta.

Essa reprovação vale para o resultado observado. Não demonstra que todo uso de Add Guide falha; o modo permanece útil como controle, e outros resultados com guides/anchors já foram aprovados pelo usuário. O caminho principal de investigação continua sendo prefixo fixo com memória visual nativa.

Os snapshots e manifests dos checkpoints anteriores permanecem intactos.
