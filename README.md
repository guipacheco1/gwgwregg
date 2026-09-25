# Tracker – Quebra por Apresentação

App web (Streamlit) que quebra, por apresentação, os PDVs que vêm sem apresentação
identificada, usando análise combinatória.

Você sobe os arquivos (ZIP, ou o arquivo de PDV + o de mercado) e ele:

1. Identifica sozinho o arquivo de **PDV** e o de **mercado**.
2. Mapeia as colunas por sinônimos (`Estado`/`UF`, `Canal`, `Fator`...); se o layout mudar,
   é possível ajustar o mapeamento na tela.
3. Calcula os **restantes**: mercado − PDVs já identificados, por Período × Mercado × Grupo × Canal × Geografia × Apresentação.
   A apresentação no arquivo de PDV é opcional: sem ela, todos os PDVs são quebrados.
4. Faz o loop por combinação no nível **UF**, **Cidade** ou **Região**.
5. Em cada combinação, resolve um modelo inteiro (HiGHS) que distribui os restantes entre os PDVs:
   unidades (sempre), valor (opcional) e número primo / fator (opcional).
6. Exporta a quebra por PDV, a matriz PDV × apresentação, o log das combinações e o arquivo de PDV quebrado.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estrutura

- `app.py` — interface
- `nucleo/leitura.py` — leitura de ZIP/TXT/CSV/XLSX, encoding, separador e mapeamento de colunas
- `nucleo/motor.py` — restantes, fator/preço médio, loop e otimização
