from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def get_quantum_encoder(num_features: int):
    """
    Creates a quantum circuit for data encoding using Angle Encoding.
    Each classical feature is encoded as a rotation around the Y-axis.
    """
    qc = QuantumCircuit(num_features)
    inputs = ParameterVector('input', num_features)
    
    # Angle encoding using RY gates
    for i in range(num_features):
        qc.ry(inputs[i], i)
        
    return qc
