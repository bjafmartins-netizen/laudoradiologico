import pulp
import pandas as pd

# ==============================================================================
# 1. MAPEAMENTO DE MÉDICOS E APTIDÕES
# ==============================================================================
medicos = [
    "alexandre", "angelo", "araci", "bernardo", "denise", "elaine", "fabio",
    "fernanda", "giovani", "graciela", "lorena", "makoto", "marcus", "martha",
    "olivia", "paulo", "paola", "thiago", "yana"
]

tc_neuro_aptos = ["alexandre", "angelo", "araci", "fabio", "fernanda", "giovani", "lorena", "makoto", "marcus", "paulo"]
tc_msk_aptos = ["bernardo", "denise", "elaine", "graciela", "martha", "olivia", "paola", "thiago", "yana"]

rm_neuro_aptos = tc_neuro_aptos
rm_msk_aptos = tc_msk_aptos

us_geral_aptos = [m for m in medicos if m not in ["fernanda", "olivia"]]
us_doppler_aptos = ["bernardo", "fabio", "giovani", "paulo", "thiago"]
us_musculo_aptos = ["denise", "elaine", "martha", "olivia", "paola", "thiago"]
us_nervo_aptos = ["graciela"]
us_inf_aptos = ["elaine"]

ee_aptos = medicos
ln_aptos = medicos
infiltracao_aptos = ["angelo", "bernardo", "fabio", "makoto", "paulo", "thiago"]
video_aptos = ["araci", "fernanda", "lorena", "yana"]
laudo_do_aptos = ["angelo", "denise", "elaine", "lorena", "yana", "fernanda"]

# ==============================================================================
# 2. CALENDÁRIO JUNHO 2026
# ==============================================================================
dias_uteis = [1, 2, 3, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30]
vesperas_feriado = [3]

turnos = ["Manha", "Tarde"]
atividades_A = ["TC_Neuro", "TC_MSK", "RM_Neuro", "RM_MSK"]
atividades_B = ["US_Geral", "US_Doppler", "US_Musculo", "US_Nervo", "US_Inf", "EE", "LN", "Infiltracao", "Video", "LaudoDO"]
todas_atividades = atividades_A + atividades_B

def obter_dia_da_semana(dia):
    # Considerando 1 de junho de 2026 como uma Segunda-feira (0)
    return (dia - 1) % 7

semanas = {
    1: [1, 2, 3, 5],
    2: [8, 9, 10, 11, 12],
    3: [15, 16, 17, 18, 19],
    4: [22, 23, 24, 25, 26],
    5: [29, 30]
}

def create_and_solve_model():
    prob = pulp.LpProblem("Escala_Radiologia_Robust", pulp.LpMinimize)

    # Use LpBinary explicit type
    X = pulp.LpVariable.dicts("X", ((m, d, t, a) for m in medicos for d in dias_uteis for t in turnos for a in todas_atividades), cat=pulp.LpBinary)

    quebra_intersticio = pulp.LpVariable.dicts("quebra_intersticio", ((m, d) for m in medicos for d in dias_uteis), lowBound=0, cat=pulp.LpBinary)

    for d in dias_uteis:
        dia_semana = obter_dia_da_semana(d)

        for t in turnos:
            for m in medicos:
                if m not in tc_neuro_aptos: prob += X[m, d, t, "TC_Neuro"] == 0
                if m not in tc_msk_aptos: prob += X[m, d, t, "TC_MSK"] == 0
                if m not in rm_neuro_aptos: prob += X[m, d, t, "RM_Neuro"] == 0
                if m not in rm_msk_aptos: prob += X[m, d, t, "RM_MSK"] == 0
                if m not in us_geral_aptos: prob += X[m, d, t, "US_Geral"] == 0
                if m not in us_doppler_aptos: prob += X[m, d, t, "US_Doppler"] == 0
                if m not in us_musculo_aptos: prob += X[m, d, t, "US_Musculo"] == 0
                if m not in us_nervo_aptos: prob += X[m, d, t, "US_Nervo"] == 0
                if m not in us_inf_aptos: prob += X[m, d, t, "US_Inf"] == 0
                if m not in ee_aptos: prob += X[m, d, t, "EE"] == 0
                if m not in ln_aptos: prob += X[m, d, t, "LN"] == 0
                if m not in infiltracao_aptos: prob += X[m, d, t, "Infiltracao"] == 0
                if m not in video_aptos: prob += X[m, d, t, "Video"] == 0
                if m not in laudo_do_aptos: prob += X[m, d, t, "LaudoDO"] == 0

            # Exact 1 TC doctor total per shift
            prob += pulp.lpSum(X[m, d, t, "TC_Neuro"] for m in medicos) + pulp.lpSum(X[m, d, t, "TC_MSK"] for m in medicos) == 1
            # Exact 1 RM doctor total per shift
            prob += pulp.lpSum(X[m, d, t, "RM_Neuro"] for m in medicos) + pulp.lpSum(X[m, d, t, "RM_MSK"] for m in medicos) == 1

            # Complementarity: if TC is Neuro, RM is MSK.
            prob += pulp.lpSum(X[m, d, t, "TC_Neuro"] for m in medicos) == pulp.lpSum(X[m, d, t, "RM_MSK"] for m in medicos)
            prob += pulp.lpSum(X[m, d, t, "TC_MSK"] for m in medicos) == pulp.lpSum(X[m, d, t, "RM_Neuro"] for m in medicos)

            # Ultrasound rules setup
            vagas_us_geral = 0; vagas_us_musculo = 0; vagas_us_doppler = 0; vagas_us_nervo = 0; vagas_us_inf = 0

            if t == "Manha":
                if dia_semana in [0, 1, 2, 3, 4]:
                    vagas_us_geral = 2
            elif t == "Tarde":
                if dia_semana in [0, 2, 3, 4]:
                    vagas_us_geral = 1
                    if dia_semana == 3 and d == 25:
                        vagas_us_inf = 1
                    elif dia_semana == 4 and d == 5:
                        vagas_us_nervo = 1
                    else:
                        vagas_us_musculo = 1
                elif dia_semana == 1:
                    vagas_us_geral = 1
                    vagas_us_doppler = 1

            prob += pulp.lpSum(X[m, d, t, "US_Geral"] for m in medicos) == vagas_us_geral
            prob += pulp.lpSum(X[m, d, t, "US_Musculo"] for m in medicos) == vagas_us_musculo
            prob += pulp.lpSum(X[m, d, t, "US_Doppler"] for m in medicos) == vagas_us_doppler
            prob += pulp.lpSum(X[m, d, t, "US_Nervo"] for m in medicos) == vagas_us_nervo
            prob += pulp.lpSum(X[m, d, t, "US_Inf"] for m in medicos) == vagas_us_inf

            # Non-accumulation of shifts (Except LaudoDO)
            for m in medicos:
                prob += pulp.lpSum(X[m, d, t, a] for a in todas_atividades if a != "LaudoDO") <= 1

    # Limite Semanal Tipo A (Soft Constraint - penalty for > 1)
    excesso_tipo_a = pulp.LpVariable.dicts("excesso_tipo_A", ((m, sem) for m in medicos for sem in semanas.keys()), lowBound=0, cat="Continuous")

    for sem_id, dias_sem in semanas.items():
        dias_validos = [d for d in dias_sem if d in dias_uteis]
        for m in medicos:
            prob += pulp.lpSum(X[m, d, t, a] for d in dias_validos for t in turnos for a in atividades_A) - 1 <= excesso_tipo_a[m, sem_id]

    # Interstício
    for i, d in enumerate(dias_uteis[:-1]):
        d_seguinte = dias_uteis[i+1]
        for m in medicos:
            prob += pulp.lpSum(X[m, d_seguinte, t, a] for t in turnos for a in todas_atividades) <= \
                    8 * (1 - pulp.lpSum(X[m, d, t, a] for t in turnos for a in atividades_A)) + 8 * quebra_intersticio[m, d]

    desvio_A = pulp.LpVariable.dicts("desvio_A", medicos, lowBound=0)
    desvio_B = pulp.LpVariable.dicts("desvio_B", medicos, lowBound=0)
    desvio_sextas = pulp.LpVariable.dicts("desvio_sex", medicos, lowBound=0)

    media_A = 42 / 19  #
    media_B = 84 / 19  #

    for m in medicos:
        total_A = pulp.lpSum(X[m, d, t, a] for d in dias_uteis for t in turnos for a in atividades_A)
        total_B = pulp.lpSum(X[m, d, t, a] for d in dias_uteis for t in turnos for a in atividades_B)
        prob += total_A - media_A <= desvio_A[m]
        prob += media_A - total_A <= desvio_A[m]
        prob += total_B - media_B <= desvio_B[m]
        prob += media_B - total_B <= desvio_B[m]

        total_sex_vesp = pulp.lpSum(X[m, d, t, a] for d in dias_uteis for t in turnos for a in atividades_A if obter_dia_da_semana(d) == 4 or d in vesperas_feriado)
        prob += total_sex_vesp - (8/19) <= desvio_sextas[m]

    preferencia_olivia = pulp.lpSum(X["olivia", d, t, a] for d in dias_uteis for t in turnos for a in atividades_A if obter_dia_da_semana(d) in [0, 2])
    evitar_plantao = pulp.lpSum(X[m, d, "Manha", a] for m in ["bernardo", "graciela", "giovani"] for d in dias_uteis for a in todas_atividades if obter_dia_da_semana(d) in [1, 3])

    prob += (pulp.lpSum(desvio_A[m] for m in medicos) * 50 +
             pulp.lpSum(desvio_B[m] for m in medicos) * 20 +
             pulp.lpSum(quebra_intersticio[m, d] for m in medicos for d in dias_uteis) * 1000 +
             pulp.lpSum(excesso_tipo_a[m, sem_id] for m in medicos for sem_id in semanas.keys()) * 200 +
             pulp.lpSum(desvio_sextas[m] for m in medicos) * 10 +
             evitar_plantao * 15 -
             preferencia_olivia * 10)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    return prob, status, X

def generate_schedule_df(status, X):
    if pulp.LpStatus[status] in ["Optimal", "Infeasible"]:
        linhas = []
        for d in sorted(dias_uteis):
            for t in turnos:
                for a in todas_atividades:
                    escalados = [m for m in medicos if X[m, d, t, a].varValue and X[m, d, t, a].varValue > 0.5]
                    for med in escalados:
                        linhas.append({
                            "Dia": f"{d:02d}/06/2026",
                            "Turno": t,
                            "Atividade": a,
                            "Médico": med.capitalize()
                        })

        df = pd.DataFrame(linhas)
        return df
    else:
        return None

if __name__ == "__main__":
    prob, status, X = create_and_solve_model()
    df = generate_schedule_df(status, X)
    if df is not None:
        print("🚀 Escala de Radiologia gerada e salva com sucesso!")
        print("\n--- AMOSTRA DA ESCALA DO INÍCIO DO MÊS ---")
        print(df.head(15).to_string(index=False))
        df.to_csv("escala_radiologia_junho_2026.csv", index=False, encoding="utf-8-sig")
    else:
        print("❌ Falha crítica no processador matemático. Status:", pulp.LpStatus[status])
