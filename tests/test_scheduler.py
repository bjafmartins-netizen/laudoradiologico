import pytest
import pandas as pd
from radiologia import scheduler

@pytest.fixture(scope="module")
def schedule_data():
    prob, status, X = scheduler.create_and_solve_model()
    df = scheduler.generate_schedule_df(status, X)
    return df

def test_schedule_generated(schedule_data):
    assert schedule_data is not None
    assert not schedule_data.empty

def test_tc_rm_coverage(schedule_data):
    grouped = schedule_data.groupby(['Dia', 'Turno'])

    for (dia, turno), group in grouped:
        atividades = group['Atividade'].value_counts()
        tc_neuro_count = atividades.get('TC_Neuro', 0)
        tc_msk_count = atividades.get('TC_MSK', 0)
        rm_neuro_count = atividades.get('RM_Neuro', 0)
        rm_msk_count = atividades.get('RM_MSK', 0)

        # Exact 1 TC doctor
        assert tc_neuro_count + tc_msk_count == 1, f"Failed TC count at {dia} {turno}"
        # Exact 1 RM doctor
        assert rm_neuro_count + rm_msk_count == 1, f"Failed RM count at {dia} {turno}"

        # Complementarity
        assert tc_neuro_count == rm_msk_count, f"Complementarity mismatch at {dia} {turno}"
        assert tc_msk_count == rm_neuro_count, f"Complementarity mismatch at {dia} {turno}"

def test_doctor_aptitude(schedule_data):
    aptitudes = {
        "TC_Neuro": scheduler.tc_neuro_aptos,
        "TC_MSK": scheduler.tc_msk_aptos,
        "RM_Neuro": scheduler.rm_neuro_aptos,
        "RM_MSK": scheduler.rm_msk_aptos,
        "US_Geral": scheduler.us_geral_aptos,
        "US_Doppler": scheduler.us_doppler_aptos,
        "US_Musculo": scheduler.us_musculo_aptos,
        "US_Nervo": scheduler.us_nervo_aptos,
        "US_Inf": scheduler.us_inf_aptos,
        "EE": scheduler.ee_aptos,
        "LN": scheduler.ln_aptos,
        "Infiltracao": scheduler.infiltracao_aptos,
        "Video": scheduler.video_aptos,
        "LaudoDO": scheduler.laudo_do_aptos,
    }

    for idx, row in schedule_data.iterrows():
        atividade = row['Atividade']
        medico = row['Médico'].lower()
        if atividade in aptitudes:
            assert medico in aptitudes[atividade], f"Doctor {medico} not apt for {atividade}"

def test_no_simultaneous_shifts(schedule_data):
    grouped = schedule_data.groupby(['Dia', 'Turno', 'Médico'])
    for (dia, turno, medico), group in grouped:
        non_laudo = group[group['Atividade'] != 'LaudoDO']
        assert len(non_laudo) <= 1, f"Doctor {medico} has multiple non-LaudoDO activities at {dia} {turno}"

def test_ultrasound_rules(schedule_data):
    grouped = schedule_data.groupby(['Dia', 'Turno'])

    for (dia, turno), group in grouped:
        atividades = group['Atividade'].value_counts()
        us_geral = atividades.get('US_Geral', 0)
        us_musculo = atividades.get('US_Musculo', 0)
        us_doppler = atividades.get('US_Doppler', 0)
        us_nervo = atividades.get('US_Nervo', 0)
        us_inf = atividades.get('US_Inf', 0)

        dia_int = int(dia.split('/')[0])
        dia_semana = scheduler.obter_dia_da_semana(dia_int)

        if turno == "Manha":
            if dia_semana in [0, 1, 2, 3, 4]:
                assert us_geral == 2
            else:
                assert us_geral == 0
        elif turno == "Tarde":
            if dia_semana in [0, 2, 3, 4]:
                assert us_geral == 1
                if dia_semana == 3 and dia_int == 25:
                    assert us_inf == 1
                    assert us_musculo == 0
                    assert us_nervo == 0
                elif dia_semana == 4 and dia_int == 5:
                    assert us_nervo == 1
                    assert us_inf == 0
                    assert us_musculo == 0
                else:
                    assert us_musculo == 1
                    assert us_nervo == 0
                    assert us_inf == 0
                assert us_doppler == 0
            elif dia_semana == 1:
                assert us_geral == 1
                assert us_doppler == 1
                assert us_musculo == 0
                assert us_nervo == 0
                assert us_inf == 0
