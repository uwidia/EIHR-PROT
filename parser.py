import gemmi

def get_protein_info(file_path):
    with open(file_path, "r") as file:
        fasta_file = file.read()

    protein_info = []
    per_protein_info = fasta_file.split(">")[1:]

    for protein in per_protein_info:
        lines = protein.splitlines()
        header = lines[0].strip()
        sequence ="".join(lines[1:])
        full_id = header.split()[0].upper()

        parts = full_id.split("-")
        entry_id = parts[0]

        chain = parts[1] if len(parts) > 1 else None

        protein_i = {
            "entry_id": entry_id, 
            "full_id" : full_id, 
            "chain": chain,
            "sequence": sequence
            }
        protein_info.append(protein_i)

    return protein_info

#Parse cif file and retreive method
def get_method(cif_path):
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    methods = block.find_values("_exptl.method")
    method = "; ".join(methods) if methods else "UNKNOWN"

    return method

#Parse retained_ids for xray-derived pdb structures
def get_non_xray_ids(file_path):
    with open(file_path, "r", encoding = "utf-8") as file:
        deleted_ids_txt = file.read()
        non_xray_id = deleted_ids_txt.splitlines()
        non_xray_id = set(non_xray_id)
    return non_xray_id
