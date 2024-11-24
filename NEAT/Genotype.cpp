#include "Genotype.h"
#include "Genes.h"
#include "Utils.h"
#include "Math.h"
#include "Map.h"
#include <sstream>
#include <string>

NEAT::InnovationTracker NEAT::Innovations = NEAT::InnovationTracker();

// Removes connections that have invalid input or output nodes
void NEAT::Genotype::Prune()
{
	Connections = Connections.FilterByPredicate(ValidConnectionFilter());
}

NEAT::Genotype::ConnectionFilter NEAT::Genotype::ValidConnectionFilter() const
{
	return [this](const auto& Connection)->bool
	{
		if (!Nodes.Contains(Connection.second.Input)) return false;
		if (!Nodes[Connection.second.Input].Enabled) return false;
		if (!Nodes.Contains(Connection.second.Output)) return false;
		if (!Nodes[Connection.second.Output].Enabled) return false;
		return true;
	};
}

/**
 * Reduces the gene keys of the genotype definition.
 *
 * This function reassigns the gene keys of the nodes and connections in the genotype definition
 * to a contiguous range starting from 0. This is done to remove any gaps in the gene key
 * sequence that may have arisen due to node or connection removal.
 */
void NEAT::Genotype::ReduceGeneKeys()
{
    uint64 NextGUID = 0;  // Initialize the next GUID to 0

    // Nodes  
    auto TempNodes = Nodes.GetValues(); // Create a temporary array to store the node genes
	auto NumNodes = TempNodes.Num(); // Get the number of nodes  
	Nodes.Reset(); // Clear the node genes map
    TempNodes.Sort([](const auto& LHS, const auto& RHS) { return LHS.ID < RHS.ID; }); // Sort the node genes by their gene ID

    for (uint64 StopIdx = NumNodes; NextGUID != StopIdx; ++NextGUID) // Iterate over the nodes and reassign their gene IDs 
    {
        uint64 GeneID = TempNodes[NextGUID].ID; // Get the current node's gene ID 
        if (NextGUID == GeneID) continue; // If the gene ID is already equal to the next GUID, skip this iteration  
        for (auto& ConnectionGene : Connections) // Iterate over the connection genes and update their input and output IDs
        {
			auto& Connection = ConnectionGene.second; // Get the current connection gene
            if (Connection.Input == GeneID) Connection.Input = NextGUID; // If the connection's input ID matches the current node's gene ID, update it  
            if (Connection.Output == GeneID) Connection.Output = NextGUID; // If the connection's output ID matches the current node's gene ID, update it  
        }
        TempNodes[NextGUID].ID = NextGUID; // Update the node's gene ID 
    }
      
    for (const auto& Node : TempNodes) Nodes[Node.ID] = Node;  // Add the updated nodes back to the node genes map

    // Connections  
    auto TempConnections = Connections.GetValues(); // Create a temporary array to store the connection genes
	int NumConnections = TempConnections.Num(); // Get the number of connections
    Connections.Reset(); // Clear the node genes map
    TempConnections.Sort([](const auto& LHS, const auto& RHS) { return LHS.ID < RHS.ID; }); // Sort the connection genes by their gene ID      
      
    for (uint64 StopIdx = NumNodes + NumConnections; NextGUID != StopIdx; ++NextGUID) // Iterate over the connections and reassign their gene IDs
    {
        uint64 ConnectionIdx = NextGUID - NumNodes; // Calculate the connection's index
        uint64 GeneKey = TempConnections[ConnectionIdx].ID; // Get the current connection's gene ID
        if (NextGUID == GeneKey) continue; // If the gene key is already equal to the next GUID, skip this iteration
        Connections[ConnectionIdx].ID = NextGUID; // Update the connection's gene ID
    }
      
    for (const auto& Connection : TempConnections) Connections[Connection.ID] = Connection; // Add the updated connections back to the connection genes map
}

/**
 * Prints the genotype definition with LogMessage
 */
void NEAT::Genotype::PrintGenotype() const
{
	LogMessage(LogLevel::Info, "Genotype:");
	LogMessage(LogLevel::Info, "Nodes:");
	for (const auto& Node : Nodes)
	{
		LogMessage(LogLevel::Info, "ID: " + std::to_string(Node.second.ID) + ", Type: " + NodeType::ToString(Node.second.Type) + ", Activation: " + Activation::ToString(Node.second.Activation) + ", Aggregation: " + Aggregation::ToString(Node.second.Aggregation) + ", Bias: " + std::to_string(Node.second.Bias));
	}
	LogMessage(LogLevel::Info, "Connections:");
	for (const auto& Connection : Connections)
	{
		LogMessage(LogLevel::Info, "ID: " + std::to_string(Connection.second.ID) + ", Input: " + std::to_string(Connection.second.Input) + ", Output: " + std::to_string(Connection.second.Output) + ", Weight: " + std::to_string(Connection.second.Weight) + ", Enabled: " + std::to_string(Connection.second.Enabled));
	}
}

/**
 * Returns the newest gene key in the genotype definition.
 *
 * @return The newest gene key in the genotype definition.
 */
uint64 NEAT::Genotype::GetNewestGeneKey() const
{
	uint64 NewestGeneKey = 0; // Initialize the newest gene key to 0
	for (const auto& Node : Nodes) NewestGeneKey = Math::Max<uint64>(NewestGeneKey, Node.second.ID); // Iterate over the nodes and update the newest gene key
	for (const auto& Connection : Connections) NewestGeneKey = Math::Max<uint64>(NewestGeneKey, Connection.second.ID); // Iterate over the connections and update the newest gene key
	return NewestGeneKey; // Return the newest gene key
}

TArray<uint64> NEAT::Genotype::GetConnectionKeys() const
{
	return Connections.GetKeys();
}

TArray<uint64> NEAT::Genotype::GetNodeKeys() const
{
	return Nodes.GetKeys();
}

std::string NEAT::Genotype::ToPrettyString() const
{
	std::stringstream ss;

	ss << "Nodes:\n";
	for (const auto& Node : Nodes)
	{
		ss << "  Node " << Node.second.ID << " (Type: " << NodeType::ToString(Node.second.Type) << ", Bias: " << Node.second.Bias << ", Aggregation: "
			<< Aggregation::ToString(Node.second.Aggregation) << ", Activation: " << Activation::ToString(Node.second.Activation) << ", Enabled: " << Node.second.Enabled << ")\n";
	}

	ss << "Connections:\n";
	for (const auto& Connection : Connections)
	{
		ss << "  Connection " << Connection.second.ID << " (From: " << Connection.second.Input << ", To: " << Connection.second.Output << ", Weight: " << Connection.second.Weight << ", Enabled: " << Connection.second.Enabled << ")\n";
	}

	return ss.str();
}

/**
 * Deserializes the genotype definition from a YAML string.
 *
 * @param Data The string containing the serialized genotype definition.
 * @return True if the genotype definition was successfully deserialized, false otherwise.
 */
bool NEAT::Genotype::Deserialize(const std::string& Data)
{
    try
    {
        // Parse the YAML string  
        std::istringstream iss(Data);
        std::string line;
        std::map<std::string, std::string> yamlMap;

        while (std::getline(iss, line)) {
            if (line.find(":") != std::string::npos) {
                size_t colonPos = line.find(":");
                std::string key = line.substr(0, colonPos);
                std::string value = line.substr(colonPos + 1);
                yamlMap[key] = value;
            }
        }

        // Deserialize the nodes  
        if (yamlMap.find("nodes") != yamlMap.end()) {
            std::string nodesStr = yamlMap["nodes"];
            std::istringstream nodesIss(nodesStr);
            std::string nodeLine;

            while (std::getline(nodesIss, nodeLine)) {
                if (nodeLine.find("-") != std::string::npos) {
                    size_t dashPos = nodeLine.find("-");
                    std::string nodeIdStr = nodeLine.substr(dashPos + 1);
                    uint64 nodeId = std::stoull(nodeIdStr);

                    // Get the node type  
                    std::string nodeTypeStr = yamlMap["nodes." + nodeIdStr + ".type"];
                    ENodeType nodeType = NodeType::FromString(nodeTypeStr);

                    // Get the node bias  
                    std::string nodeBiasStr = yamlMap["nodes." + nodeIdStr + ".bias"];
                    double nodeBias = std::stod(nodeBiasStr);

                    // Get the node aggregation function  
                    std::string nodeAggregationStr = yamlMap["nodes." + nodeIdStr + ".aggregation"];
                    EAggregation nodeAggregation = Aggregation::FromString(nodeAggregationStr);

                    // Get the node activation function  
                    std::string nodeActivationStr = yamlMap["nodes." + nodeIdStr + ".activation"];
                    EActivation nodeActivation = Activation::FromString(nodeActivationStr);

                    // Get the node enabled flag  
                    std::string nodeEnabledStr = yamlMap["nodes." + nodeIdStr + ".enabled"];
                    bool nodeEnabled = (nodeEnabledStr == "true");

                    // Add the node gene to the node genes map  
                    Nodes[nodeId] = NEAT::NodeGene(nodeId, nodeType, nodeActivation, nodeAggregation, nodeBias, nodeEnabled);
                }
            }
        }

        // Deserialize the connections  
        if (yamlMap.find("connections") != yamlMap.end()) {
            std::string connectionsStr = yamlMap["connections"];
            std::istringstream connectionsIss(connectionsStr);
            std::string connectionLine;

            while (std::getline(connectionsIss, connectionLine)) {
                if (connectionLine.find("-") != std::string::npos) {
                    size_t dashPos = connectionLine.find("-");
                    std::string connectionIdStr = connectionLine.substr(dashPos + 1);
                    uint64 connectionId = std::stoull(connectionIdStr);

                    // Get the connection source node ID  
                    std::string connectionSourceStr = yamlMap["connections." + connectionIdStr + ".source"];
                    uint64 connectionSource = std::stoull(connectionSourceStr);

                    // Get the connection target node ID  
                    std::string connectionTargetStr = yamlMap["connections." + connectionIdStr + ".target"];
                    uint64 connectionTarget = std::stoull(connectionTargetStr);

                    // Get the connection weight  
                    std::string connectionWeightStr = yamlMap["connections." + connectionIdStr + ".weight"];
                    double connectionWeight = std::stod(connectionWeightStr);

                    // Get the connection enabled flag  
                    std::string connectionEnabledStr = yamlMap["connections." + connectionIdStr + ".enabled"];
                    bool connectionEnabled = (connectionEnabledStr == "true");

                    // Add the connection gene to the connection genes map  
                    Connections[connectionId] = NEAT::ConnectionGene(connectionId, connectionSource, connectionTarget, connectionWeight, connectionEnabled);
                }
            }
        }

        return true; // Return true to indicate success  
    }
    catch (const std::exception& e) // Catch any exceptions  
    {
        LogMessage(LogLevel::Error, "Failed to deserialize genotype: " + std::string(e.what())); // Log an error message  
        return false; // Return false to indicate failure  
    }
}

/**
 * Serializes the genotype definition to a YAML string.
 *
 * @return The serialized genotype definition.
 */
std::string NEAT::Genotype::Serialize()
{
    std::string yamlStr;

    // Serialize the nodes  
    yamlStr += "nodes:\n";
    for (const auto& node : Nodes) {
        yamlStr += "  - " + std::to_string(node.first) + ":\n";
        yamlStr += "    type: " + NodeType::ToString(node.second.Type) + "\n";
        yamlStr += "    bias: " + std::to_string(node.second.Bias) + "\n";
        yamlStr += "    aggregation: " + Aggregation::ToString(node.second.Aggregation) + "\n";
        yamlStr += "    activation: " + Activation::ToString(node.second.Activation) + "\n";
        yamlStr += "    enabled: " + std::string(node.second.Enabled ? "true" : "false") + "\n";
    }

    // Serialize the connections  
    yamlStr += "connections:\n";
    for (const auto& connection : Connections) {
        yamlStr += "  - " + std::to_string(connection.first) + ":\n";
        yamlStr += "    source: " + std::to_string(connection.second.Input) + "\n";
        yamlStr += "    target: " + std::to_string(connection.second.Output) + "\n";
        yamlStr += "    weight: " + std::to_string(connection.second.Weight) + "\n";
        yamlStr += "    enabled: " + std::string(connection.second.Enabled ? "true" : "false") + "\n";
    }

    return yamlStr; // Return the serialized genotype definition  
}

void NEAT::Genotype::Mutate(const ConfigPtr& Config)
{
	if (Config->SingleMutation)
	{
		// Choose a random mutation
		switch (static_cast<EMutationType>(rand() % static_cast<int>(EMutationType::MAX)))
		{
		case EMutationType::AddNode: if (Math::Random<double>(1.0) < Config->AddNodeMutationRate) MutateAddNode(Config); break;
		case EMutationType::AddConnection: if (Math::Random<double>(1.0) < Config->AddConnectionMutationRate) MutateAddConnection(Config); break;
		case EMutationType::RemoveNode: if (Math::Random<double>(1.0) < Config->RemoveNodeMutationRate) MutateRemoveNode(Config); break;
		case EMutationType::RemoveConnection: if (Math::Random<double>(1.0) < Config->RemoveConnectionMutationRate) MutateRemoveConnection(Config); break;
		case EMutationType::ModifyWeight: if (Math::Random<double>(1.0) < Config->WeightMutationRate) MutateModifyWeight(Config); break;
		case EMutationType::ModifyBias: if (Math::Random<double>(1.0) < Config->BiasMutationRate) MutateModifyBias(Config); break;
		case EMutationType::ModifyActivation: if (Math::Random<double>(1.0) < Config->ActivationFunctionMutationRate) MutateModifyActivation(Config); break;
		case EMutationType::ModifyAggregation: if (Math::Random<double>(1.0) < Config->AggregationFunctionMutationRate) MutateModifyAggregation(Config); break;
		case EMutationType::ToggleConnection: if (Math::Random<double>(1.0) < Config->EnableMutationRate) MutateToggleConnection(Config); break;
		}
	}
	else
	{
		// Choose a random mutation
		if (Math::Random<double>(1.0) < Config->AddNodeMutationRate) MutateAddNode(Config);
		if (Math::Random<double>(1.0) < Config->AddConnectionMutationRate) MutateAddConnection(Config);
		if (Math::Random<double>(1.0) < Config->RemoveNodeMutationRate) MutateRemoveNode(Config);
		if (Math::Random<double>(1.0) < Config->RemoveConnectionMutationRate) MutateRemoveConnection(Config);
		if (Math::Random<double>(1.0) < Config->WeightMutationRate) MutateModifyWeight(Config);
		if (Math::Random<double>(1.0) < Config->BiasMutationRate) MutateModifyBias(Config);
		if (Math::Random<double>(1.0) < Config->ActivationFunctionMutationRate) MutateModifyActivation(Config);
		if (Math::Random<double>(1.0) < Config->AggregationFunctionMutationRate) MutateModifyAggregation(Config);
		if (Math::Random<double>(1.0) < Config->EnableMutationRate) MutateToggleConnection(Config);
	}
}

bool NEAT::Genotype::MutateAddNode(const NEAT::ConfigPtr& Config)
{
	if (Connections.IsEmpty()) return false; // No connections to split
	const auto ConnectionID = Connections.GetKeys()[rand() % Connections.Num()]; // Find a random connection ID
	auto& Connection = Connections[ConnectionID]; // Find the connection
	Connection.Enabled = false; // Disable the old connection
	auto NodeID = Innovations.GetInnovationID(EMutationType::AddNode, EGeneType::Node, Connection.Input, Connection.Output); // Get the new Node ID
	auto InputID = Innovations.GetInnovationID(EMutationType::AddNode, EGeneType::Connection, Connection.Input, NodeID); // Get the ID for the connection from old input to new node
	auto OutputID = Innovations.GetInnovationID(EMutationType::AddNode, EGeneType::Connection, NodeID, Connection.Output); // Get the ID for the connection from new node to old output
	if (Nodes.Contains(NodeID)) return false; // Node already exists
    Nodes[NodeID] = NodeGene(NodeID, ENodeType::Hidden, Config->DefaultActivationFunction, Config->DefaultAggregationFunction, 0.0); // Create a new node
	Connections[InputID] = ConnectionGene(InputID, Connection.Input, NodeID, 1.0); // Create a new connection from the old source node to the new node
	Connections[OutputID] = ConnectionGene(OutputID, NodeID, Connection.Output, Connection.Weight); // Create a new connection from the new node to the old target node
    return true;
}

bool NEAT::Genotype::MutateAddConnection(const NEAT::ConfigPtr& Config)
{
	const auto& NodesIDs = Nodes.GetKeys(); // Get all node IDs
	if (NodesIDs.IsEmpty()) return false; // No nodes to connect
	auto Node1ID = NodesIDs[rand() % NodesIDs.Num()]; // Find first random node
	while (Nodes[Node1ID].Type == ENodeType::Output) Node1ID = NodesIDs[rand() % NodesIDs.Num()]; // Don't connect output to anything
	auto Node2ID = NodesIDs[rand() % NodesIDs.Num()]; // Find second random node
	while (Nodes[Node2ID].Type == ENodeType::Input) Node2ID = NodesIDs[rand() % NodesIDs.Num()]; // Don't connect anything to input
	auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, Node1ID, Node2ID); // Get the new connection ID
	if (Connections.Contains(ConnectionID)) return false; // Connection already exists
	Connections[ConnectionID] = ConnectionGene(ConnectionID, Node1ID, Node2ID, 1.0); // Create a new connection
	return true;
}

bool NEAT::Genotype::MutateRemoveNode(const NEAT::ConfigPtr& Config)
{
	auto HiddenNodeKeys = GetFilteredNodeKeys([](const auto& Node) { return Node.second.Type != ENodeType::Input && Node.second.Type != ENodeType::Output; });
	if (HiddenNodeKeys.IsEmpty()) return false; // No hidden nodes to remove
	auto NodeID = HiddenNodeKeys[rand() % HiddenNodeKeys.Num()]; // Get random hidden node
	Nodes.Remove(NodeID); // Remove the node
	Prune(); // Remove invalid connections
	return true;
}

bool NEAT::Genotype::MutateRemoveConnection(const NEAT::ConfigPtr& Config)
{
	if (Connections.IsEmpty()) return false; // No connections to remove
	auto ConnectionID = Connections.GetKeys()[rand() % Connections.Num()]; // Get random connection
	Connections.Remove(ConnectionID); // Remove the connection
	return true;
}

bool NEAT::Genotype::MutateModifyWeight(const NEAT::ConfigPtr& Config)
{
	if (Connections.IsEmpty()) return false; // No connections to modify
	auto ConnectionID = Connections.GetKeys()[rand() % Connections.Num()]; // Get random connection
	auto& Connection = Connections[ConnectionID]; // Get the connection
	Connection.Weight = Math::Clamp(Connection.Weight + GetRandomDouble(-Config->WeightMutationVariance, Config->WeightMutationVariance), Config->MinConnectionWeight, Config->MaxConnectionWeight); // Modify the connection weight
	return true;
}

bool NEAT::Genotype::MutateModifyBias(const NEAT::ConfigPtr& Config)
{
	//auto HiddenNodeKeys = GetFilteredNodeKeys([](const auto& Node) { return Node.second.Type != ENodeType::Input && Node.second.Type != ENodeType::Output; });
	//if (HiddenNodeKeys.IsEmpty()) return false; // No hidden nodes to modify
	//auto NodeID = HiddenNodeKeys[rand() % HiddenNodeKeys.Num()]; // Get random hidden node
	auto NodeID = Nodes.GetKeys()[rand() % Nodes.Num()]; // Get random node
	auto& Node = Nodes[NodeID]; // Get the node
	Node.Bias = Math::Clamp(Node.Bias + GetRandomDouble(-Config->BiasMutationVariance, Config->BiasMutationVariance), Config->MinNodeBias, Config->MaxNodeBias); // Modify the node bias
	return false;
}

bool NEAT::Genotype::MutateModifyActivation(const NEAT::ConfigPtr& Config)
{
	TArray<EActivation> SupportedActivations = Config->SupportedActivationFunctions;
	SupportedActivations.AddUnique(Config->DefaultActivationFunction); // Add the default activation function, so there is at least always that to select from
	auto HiddenNodeKeys = GetFilteredNodeKeys([](const auto& Node) { return Node.second.Type != ENodeType::Input; });
	if (HiddenNodeKeys.IsEmpty()) return false; // No hidden nodes to modify
	auto NodeID = HiddenNodeKeys[rand() % HiddenNodeKeys.Num()]; // Get random hidden node
	auto& Node = Nodes[NodeID]; // Get the node
	Node.Activation = SupportedActivations[rand() % SupportedActivations.Num()]; // Modify the node activation function
	return false;
}

bool NEAT::Genotype::MutateModifyAggregation(const NEAT::ConfigPtr& Config)
{
	TArray<EAggregation> SupportedAggregations = Config->SupportedAggregationFunctions;
	SupportedAggregations.AddUnique(Config->DefaultAggregationFunction); // Add the default aggregation function, so there is at least always that to select from
	auto HiddenNodeKeys = GetFilteredNodeKeys([](const auto& Node) { return Node.second.Type != ENodeType::Input; });
	if (HiddenNodeKeys.IsEmpty()) return false; // No hidden nodes to modify
	auto NodeID = HiddenNodeKeys[rand() % HiddenNodeKeys.Num()]; // Get random hidden node
	auto& Node = Nodes[NodeID]; // Get the node
	Node.Aggregation = SupportedAggregations[rand() % SupportedAggregations.Num()]; // Modify the node aggregation function
	return false;
}

bool NEAT::Genotype::MutateToggleConnection(const NEAT::ConfigPtr& Config)
{
	if (Connections.IsEmpty()) return false; // No connections to toggle
	auto ConnectionID = Connections.GetKeys()[rand() % Connections.Num()]; // Get random connection
	auto& Connection = Connections[ConnectionID]; // Get the connection
	Connection.Enabled = !Connection.Enabled; // Toggle the connection
	return true;
}
