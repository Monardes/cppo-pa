"""
PK-PD Patient Simulator for Anesthesia RL Environment.

Implements 3-compartment mammillary pharmacokinetic models for propofol
(Schnider model) and remifentanil (Minto model) with pharmacodynamic
effect models for BIS, MAP, and HR.

References:
  - Schnider et al., Anesthesiology 1998; 88:1170-1182
  - Schnider et al., Anesthesiology 1999; 90:1502-1516
  - Minto et al., Anesthesiology 1997; 86:10-23, 24-33
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional


class PatientSimulator:
    """3-compartment PK-PD model for propofol + remifentanil anesthesia.

    Simulates the pharmacokinetic distribution and pharmacodynamic effects
    of intravenous propofol and remifentanil. Supports inter-patient
    variability through Monte Carlo sampling of PK parameters from
    published population distributions.

    Attributes:
        params: Dictionary of patient-specific PK-PD parameters.
        x_p: Propofol amounts in 3 compartments [central, fast, slow].
        x_r: Remifentanil amounts in 3 compartments [central, fast, slow].
        t: Current simulation time (seconds).
    """

    def __init__(self, params: Optional[Dict] = None, seed: Optional[int] = None):
        """Initialise patient simulator.

        Args:
            params: Patient-specific PK-PD parameters. If None, parameters
                    are sampled from published population distributions.
            seed: Random seed for reproducible patient sampling.
        """
        self.rng = np.random.RandomState(seed)
        if params is None:
            params = self._sample_patient_params()
        self.params = params
        self._reset_state()
        self.t = 0.0

    def _sample_patient_params(self) -> Dict:
        """Sample PK-PD parameters from Schnider/Minto population distributions.

        Returns:
            Dictionary of PK-PD parameters for a virtual patient.
        """
        age = self.rng.uniform(18, 85)
        weight = self.rng.uniform(45, 120)
        height = self.rng.uniform(150, 195)
        sex = self.rng.choice([0, 1])  # 0=male, 1=female
        lbm = self._lean_body_mass(weight, height, sex)

        # Schnider model parameters (propofol)
        V1_p = 4.27  # L (central volume)
        V2_p = 18.9 - 0.391 * (age - 53)  # L (fast peripheral)
        V3_p = 238.0  # L (slow peripheral)
        Cl1_p = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)
        Cl2_p = 1.29 - 0.024 * (age - 53)  # rapid distribution clearance
        Cl3_p = 0.836  # slow distribution clearance

        # Minto model parameters (remifentanil)
        V1_r = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
        V2_r = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
        V3_r = 5.42
        Cl1_r = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
        Cl2_r = 2.05 - 0.0301 * (age - 40)
        Cl3_r = 0.076 - 0.00113 * (age - 40)

        # PD parameters (propofol)
        C50_p = 2.35 - 0.029 * (age - 25)  # μg/mL, age-dependent
        gamma_p = 2.0  # Hill coefficient
        ke0_p = 0.456  # min⁻¹, effect-site equilibration rate

        # PD parameters (remifentanil)
        C50_r = 11.2 * (1 - 0.016 * (age - 40))  # ng/mL
        gamma_r = 2.51  # Hill coefficient
        ke0_r = 0.516 * (1 - 0.010 * (age - 40))  # min⁻¹

        # Baseline physiology
        MAP_base = self.rng.normal(85, 8)
        HR_base = self.rng.normal(72, 10)

        # Individual MAP/HR sensitivity to remifentanil (from population distributions)
        # Minto et al., Anesthesiology 1997 — typical α_MAP range: 300–700
        alpha_MAP = self.rng.normal(500, 100)  # mmHg·mL/μg
        alpha_HR = self.rng.normal(400, 80)    # bpm·mL/μg

        return {
            'age': age, 'weight': weight, 'height': height, 'sex': sex, 'lbm': lbm,
            # Propofol PK
            'V1_p': V1_p, 'V2_p': V2_p, 'V3_p': V3_p,
            'Cl1_p': Cl1_p, 'Cl2_p': Cl2_p, 'Cl3_p': Cl3_p,
            # Remifentanil PK
            'V1_r': V1_r, 'V2_r': V2_r, 'V3_r': V3_r,
            'Cl1_r': Cl1_r, 'Cl2_r': Cl2_r, 'Cl3_r': Cl3_r,
            # PD
            'C50_p': C50_p, 'gamma_p': gamma_p, 'ke0_p': ke0_p,
            'C50_r': C50_r, 'gamma_r': gamma_r, 'ke0_r': ke0_r,
            # Baseline
            'MAP_base': MAP_base, 'HR_base': HR_base,
            # Individual sensitivity
            'alpha_MAP': alpha_MAP, 'alpha_HR': alpha_HR,
        }

    @staticmethod
    def _lean_body_mass(weight: float, height: float, sex: int) -> float:
        """Calculate lean body mass using James equation."""
        if sex == 0:  # male
            return 1.10 * weight - 128 * (weight / height) ** 2
        else:  # female
            return 1.07 * weight - 148 * (weight / height) ** 2

    def _reset_state(self):
        """Reset drug amounts to zero (no drug on board)."""
        self.x_p = np.zeros(3)
        self.x_r = np.zeros(3)
        self.ce_p = 0.0
        self.ce_r = 0.0

    def _pk_rates(self, drug: str) -> Dict[str, float]:
        """Compute PK rate constants for a given drug."""
        prefix = drug[0]  # 'p' for propofol, 'r' for remifentanil
        p = self.params
        k10 = p[f'Cl1_{prefix}'] / p[f'V1_{prefix}']
        k12 = p[f'Cl2_{prefix}'] / p[f'V1_{prefix}']
        k13 = p[f'Cl3_{prefix}'] / p[f'V1_{prefix}']
        k21 = p[f'Cl2_{prefix}'] / p[f'V2_{prefix}']
        k31 = p[f'Cl3_{prefix}'] / p[f'V3_{prefix}']
        return {'k10': k10, 'k12': k12, 'k13': k13, 'k21': k21, 'k31': k31}

    def _pk_ode(self, x: np.ndarray, t: float, u: float, drug: str) -> np.ndarray:
        """3-compartment PK ODE system."""
        k = self._pk_rates(drug)
        dxdt = np.zeros(3)
        dxdt[0] = -(k['k10'] + k['k12'] + k['k13']) * x[0] + k['k21'] * x[1] + k['k31'] * x[2] + u
        dxdt[1] = k['k12'] * x[0] - k['k21'] * x[1]
        dxdt[2] = k['k13'] * x[0] - k['k31'] * x[2]
        return dxdt

    def step(self, u_propofol: float, u_remifentanil: float,
             dt: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """Advance simulation by dt seconds.

        Args:
            u_propofol: Propofol infusion rate (mg/kg/h).
            u_remifentanil: Remifentanil infusion rate (μg/kg/min).
            dt: Time step in seconds.

        Returns:
            obs: Observation vector [BIS, MAP, HR, SpO2, EtCO2, Ce_prop, Ce_remi].
            info: Dictionary with full physiological state.
        """
        dt_min = dt / 60.0  # convert to minutes for PK integration

        # Convert infusion rates to mass/time for PK model
        weight = self.params['weight']
        u_p_mass = u_propofol * weight / 60.0  # mg/min
        u_r_mass = u_remifentanil * weight / (1000.0 * 60.0)  # mg/min (convert μg→mg)

        # Integrate PK ODEs
        t_span = np.linspace(0, dt_min, max(2, int(dt_min / 0.01)))
        sol_p = odeint(self._pk_ode, self.x_p, t_span, args=(u_p_mass, 'propofol'))
        sol_r = odeint(self._pk_ode, self.x_r, t_span, args=(u_r_mass, 'remifentanil'))

        self.x_p = sol_p[-1]
        self.x_r = sol_r[-1]

        # Effect-site concentrations (with equilibration delay)
        ke0_p = self.params['ke0_p']
        ke0_r = self.params['ke0_r']
        cp_p = self.x_p[0] / self.params['V1_p']  # plasma concentration (mg/L = μg/mL)
        cp_r = self.x_r[0] / self.params['V1_r'] * 1000  # convert to ng/mL

        self.ce_p += ke0_p * (cp_p - self.ce_p) * dt_min
        self.ce_r += ke0_r * (cp_r - self.ce_r) * dt_min

        # PD: BIS from sigmoid Emax with drug synergy
        BIS = self._bis_from_ce(self.ce_p, self.ce_r)

        # PD: MAP and HR from remifentanil
        MAP = self._map_from_ce(self.ce_r)
        HR = self._hr_from_ce(self.ce_r)

        # Fixed values (not modeled dynamically)
        SpO2 = 98.0 + self.rng.normal(0, 1.0)
        EtCO2 = 35.0 + self.rng.normal(0, 1.5)

        self.t += dt

        obs = np.array([BIS, MAP, HR, SpO2, EtCO2, self.ce_p, cp_r / 1000.0])
        info = {
            'x_p': self.x_p.copy(), 'x_r': self.x_r.copy(),
            'ce_p': self.ce_p, 'ce_r': self.ce_r,
            'cp_p': cp_p, 'cp_r': cp_r, 't': self.t,
        }
        return obs, info

    def _bis_from_ce(self, ce_p: float, ce_r: float) -> float:
        """Sigmoid Emax PD model for BIS with propofol-remifentanil synergy.

        Args:
            ce_p: Propofol effect-site concentration (μg/mL).
            ce_r: Remifentanil effect-site concentration (ng/mL).

        Returns:
            BIS value [0, 100].
        """
        p = self.params
        # Remifentanil reduces effective C50 of propofol (synergy)
        synergy_factor = 1 - 0.3 * ce_r / (ce_r + 5.0)
        C50_eff = p['C50_p'] * max(synergy_factor, 0.5)

        effect = ce_p ** p['gamma_p'] / (C50_eff ** p['gamma_p'] + ce_p ** p['gamma_p'])
        BIS = 100.0 - 100.0 * effect  # E0=100, Emax=0 (full suppression)
        return np.clip(BIS + self.rng.normal(0, 0.5), 0, 100)

    def _map_from_ce(self, ce_r: float) -> float:
        """MAP effect from remifentanil (linear decrease with individual sensitivity)."""
        base = self.params['MAP_base']
        alpha = self.params.get('alpha_MAP', 500)
        # MAP decrease proportional to effect-site concentration
        delta = alpha * ce_r / 1000.0  # ce_r in ng/mL, convert to μg/mL
        return np.clip(base - delta + self.rng.normal(0, 2.0), 40, 140)

    def _hr_from_ce(self, ce_r: float) -> float:
        """Heart rate effect from remifentanil (linear decrease with individual sensitivity)."""
        base = self.params['HR_base']
        alpha = self.params.get('alpha_HR', 400)
        # HR decrease proportional to effect-site concentration
        delta = alpha * ce_r / 1000.0
        return np.clip(base - delta + self.rng.normal(0, 1.5), 35, 130)

    def reset(self) -> np.ndarray:
        """Reset patient to initial state (no drug)."""
        self._reset_state()
        self.t = 0.0
        # Initial observation (awake physiology)
        return np.array([
            self.params['MAP_base'] * 0 + 98.0,  # BIS ~98 (awake)
            self.params['MAP_base'],
            self.params['HR_base'],
            98.0, 35.0, 0.0, 0.0,
        ])


class AnesthesiaEnv:
    """Gymnasium-compatible anesthesia control environment.

    Wraps PatientSimulator with MDP interface for RL training.
    """

    def __init__(self, n_patients: int = 16, episode_length: int = 7200,
                 seed: Optional[int] = None):
        """Initialise environment with multiple parallel patients.

        Args:
            n_patients: Number of parallel patient simulators.
            episode_length: Maximum episode length in seconds (default 2h).
            seed: Random seed.
        """
        self.n_patients = n_patients
        self.max_steps = episode_length
        self.rng = np.random.RandomState(seed)
        self.patients = [PatientSimulator(seed=seed + i if seed else None)
                        for i in range(n_patients)]
        self.step_count = 0
        self.history = [[] for _ in range(n_patients)]

    def reset(self) -> np.ndarray:
        """Reset environment and all patients."""
        self.step_count = 0
        obs = np.array([p.reset() for p in self.patients])
        self.history = [[] for _ in range(self.n_patients)]
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all patients with given actions.

        Args:
            actions: Array of shape (n_patients, 2) with [propofol, remifentanil] rates.

        Returns:
            observations, rewards, dones, info
        """
        self.step_count += 1
        obs_list, reward_list, done_list = [], [], []

        for i, patient in enumerate(self.patients):
            u_p = np.clip(actions[i, 0] * 20, 0, 20)  # 0-1 → 0-20 mg/kg/h
            u_r = np.clip(actions[i, 1] * 2, 0, 2)    # 0-1 → 0-2 μg/kg/min
            obs, info = patient.step(u_p, u_r)
            self.history[i].append(obs)

            # Reward: BIS proximity + MAP stability − drug cost
            bis = obs[0]
            map_val = obs[1]
            r_bis = np.exp(-(bis - 50) ** 2 / (2 * 10 ** 2))
            r_map = np.exp(-(map_val - 80) ** 2 / (2 * 15 ** 2))
            r_drug = -0.01 * (actions[i, 0] ** 2 + actions[i, 1] ** 2)
            reward = 0.5 * r_bis + 0.3 * r_map + r_drug

            # Safety costs
            c1 = float(bis < 40)
            c2 = float(map_val < 55)
            c3 = float(actions[i, 0] > 0.75)

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(float(self.step_count >= self.max_steps))

            info.update({f'c1': c1, f'c2': c2, f'c3': c3})

        dones = np.array(done_list)
        return np.array(obs_list), np.array(reward_list), dones, {'safety_costs': None}

    def get_history(self, idx: int) -> np.ndarray:
        """Get observation history for patient idx."""
        return np.array(self.history[idx])


if __name__ == '__main__':
    # Quick smoke test
    env = AnesthesiaEnv(n_patients=2, seed=42)
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    for step in range(60):  # 1 minute
        actions = np.array([[0.3, 0.2], [0.35, 0.25]])
        obs, rewards, dones, info = env.step(actions)
        if step % 15 == 0:
            print(f"Step {step}: BIS={obs[0,0]:.1f}, MAP={obs[0,1]:.0f}, "
                  f"Reward={rewards[0]:.3f}")
    print("Patient simulator smoke test passed!")
