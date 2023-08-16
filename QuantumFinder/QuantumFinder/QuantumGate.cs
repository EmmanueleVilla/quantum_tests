using System;
namespace QuantumFinder
{
	public class QuantumGate
	{
		public String description;
		public int[] operation;

        public QuantumGate(String description, int[] operation)
		{
			this.description = description;
			this.operation = operation;
		}

        public override string ToString()
        {
            return description;
        }
    }
}

