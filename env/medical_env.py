import random
import numpy as np
from gymnasium import Env, spaces
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_terms(text: str):
    if text is None:
        return set()
    out = set()
    for tok in str(text).lower().replace("/", " ").replace("-", " ").split():
        tok = tok.strip().strip(",.;:()[]{}'\"")
        if len(tok) > 1:
            out.add(tok)
    return out


class MedicalRecommendationEnv(Env):
    def __init__(
        self,
        patient_records,
        med_list,
        gnn_embeddings,
        node_map,
        reference_rules,
        gnn_confidence_scores,
        full_population,
    ):
        super().__init__()
        self.patient_records = patient_records
        self.med_list = sorted(med_list)
        self.embeddings = gnn_embeddings
        self.node_map = node_map
        self.reference_rules = reference_rules
        self.confidence = gnn_confidence_scores
        self.full_population = full_population

        self.action_space = spaces.Discrete(len(self.med_list))

        emb_dim = int(self.embeddings.shape[1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(emb_dim,), dtype=np.float32
        )

        self.rlgmo_ids = list(self.patient_records.keys())
        self.current_rlgmo_id = None
        self.current_embedding = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_rlgmo_id = random.choice(self.rlgmo_ids)
        idx = self.node_map.get(self.current_rlgmo_id)

        if idx is None:
            self.current_embedding = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self.current_embedding = self.embeddings[idx].detach().cpu().numpy().astype(np.float32)

        return self.current_embedding, {}

    def step(self, action):
        selected_med = self.med_list[int(action)]
        rec = self.patient_records[self.current_rlgmo_id]

        diseases = set(map(str.lower, rec.get("Disease", [])))
        symptoms = set(map(str.lower, rec.get("Symptoms", [])))
        actual_meds = set(map(str.lower, rec.get("Medication", [])))

        info = {
            "patient_id": self.current_rlgmo_id,
            "selected_medication": selected_med,
            "actual_medication": list(actual_meds),
            "reward": 0.0,
            "matched_patient_id": None,
            "similarity_score": None,
            "matched_terms": [],
            "reference_match": False,
            "matched_diagnosis_terms": [],
            "reference_id": None,
            "reference_treatment": None,
            "cosine_similarity": None,
        }

        # exact hit
        if selected_med in actual_meds:
            reward = 1.0
            info["reward"] = reward
            return self.current_embedding, reward, True, False, info

        idx = self.node_map.get(self.current_rlgmo_id)
        if idx is None:
            reward = -1.0
            info["reward"] = reward
            return self.current_embedding, reward, True, False, info

        pvec = self.embeddings[idx].unsqueeze(0)
        best_sim = -1.0
        best_match = None
        best_terms = []

        for rid, other in self.full_population.items():
            if rid == self.current_rlgmo_id:
                continue
            oidx = self.node_map.get(rid)
            if oidx is None:
                continue

            od = set(map(str.lower, other.get("Disease", [])))
            osy = set(map(str.lower, other.get("Symptoms", [])))
            omeds = set(map(str.lower, other.get("Medication", [])))

            shared_d = diseases & od
            shared_s = symptoms & osy

            if (shared_d or shared_s) and (selected_med in omeds):
                ovec = self.embeddings[oidx].unsqueeze(0)
                sim = float(cosine_similarity(pvec, ovec)[0][0])
                if sim > best_sim:
                    best_sim = sim
                    best_match = rid
                    best_terms = sorted(list(shared_d | shared_s))

        guideline_hit = False
        matched_diag_terms = []
        ref_id = None
        ref_treatment = None

        for diagnose_terms, pid_list, did in self.reference_rules:
            dset = diagnose_terms if isinstance(diagnose_terms, set) else tokenize_terms(diagnose_terms)
            overlap = dset & (diseases | symptoms)
            if overlap:
                guideline_hit = True
                matched_diag_terms = sorted(list(overlap))
                ref_id = did
                ref_treatment = pid_list[0] if isinstance(pid_list, list) and pid_list else str(pid_list)
                break

        if best_match is not None and best_sim > 0:
            reward = 0.5 * best_sim
            info["matched_patient_id"] = best_match
            info["similarity_score"] = best_sim
            info["cosine_similarity"] = best_sim
            info["matched_terms"] = best_terms
        else:
            reward = -0.5

        if guideline_hit:
            reward += 0.2
            info["reference_match"] = True
            info["matched_diagnosis_terms"] = matched_diag_terms
            info["reference_id"] = ref_id
            info["reference_treatment"] = ref_treatment

        info["reward"] = float(reward)
        return self.current_embedding, float(reward), True, False, info
