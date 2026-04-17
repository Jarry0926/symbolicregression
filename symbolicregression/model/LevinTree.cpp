#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/funcitonal.h>

namespace py = pybind11;

struct Node
{
    py::object state;
    float      depth;
    float      logProbability;
    float      cost;
    Node*      parent;
}; // struct Node

struct LevinTree
{
    py::object model;
    py::list   actionList;
    Node*      nodePool;
    size_t     nodePoolSize;
    Node**     openList;
    size_t     openListSize;
}; // struct LevinTree

static void
levinTreeOpenListPush(LevinTree* const p_levinTree,
                      const Node*      p_node) noexcept
{
    p_levinTree->openList[++p_levinTree->openListSize] = p_node;
    uint32_t i = p_levinTree->openListSize;
    for (uint32_t j; // parent index
        p_levinTree->openList[j = i >> 1]->cost > p_node->cost;
        i = j)
    {
        p_levinTree->openList[i] = p_levinTree->openList[j];
        if (j == 1) {
            i = 1;
            break;
        }
    }
    p_levinTree->openList[i] = p_node;
}

static void
levinTreeOpenListPop(LevinTree* const p_levinTree,
                     const Node*      p_node) noexcept
{
    Node* node = p_levinTree->openList[p_levinTree->openListSize--];
    uint32_t i = p_levinTree->openListSize >> 1; // indices of last non-leaf node
    uint32_t j = p_node - p_levinTree->openList; // index of the popped node
    while (j <= i) {
        uint32_t l = j << 1;
        uint32_t r = l | 1;
        float minChildCost = p_levinTree->openList[l]->cost < p_levinTree->openList[r]->cost
                           ? p_levinTree->openList[l]->cost : p_levinTree->openList[l = r]->cost;
        if (minChildCost < node->cost) {
            p_levinTree->openList[j] = p_levinTree->openList[l];
            j = l;
        }
        else break;
    }
    p_levinTreethis->openList[j] = node;
}

void
LevinTreeCreate(LevinTree**       p_levinTree,
                const py::object  p_model,
                const py::list    p_actionList,
                const py::object  p_initialState,
                const size_t      p_treeSize) noexcept
{
    (*p_levinTree)->model        = p_model;

    (*p_levinTree)->actionList   = p_actionList;

    (*p_levinTree)->nodePool     = new node[p_treeSize * sizeof(Node)];
    (*p_levinTree)->nodePool[0]  = {p_initialState, 1.0f, 0.0f, 0.0f, nullptr};

    (*p_levinTree)->nodePoolSize = 1;

    (*p_levinTree)->openList     = new node*[(p_treeSize + 1) * sizeof(Node*)];
    (*p_levinTree)->openList[1]  = (*p_levinTree)->nodePool;

    (*p_levinTree)->openListSize = 1;
}

void
LevinTreeDestroy(LevinTree* const p_levinTree) noexcept
{
    if (p_leviTree->nodePool != nullptr) {
        delete[] p_levinTree->nodePool;
        p_levinTree->nodePool = nullptr;
    }
    
    if (p_levinTree->openList != nullptr) {
        delete[] p_levinTree->openList;
        p_levinTree->openList = nullptr;
    }
}

py::object
LevinTreeSearch(LevinTree* const   p_levinTree,
                const py::function p_getPolicyFn,
                const py::function p_applyActionFn,
                const py::function p_isSolutionFn) noexcept
{
    while (p_levinTree->openListSize > 0) {
        const Node* thisNodePtr = p_levinTree->openList[1]; // node to be expanded
        levinTreeOpenListPop(p_levinTree, thisNodePtr);
        // TODO: get policy list here
        for (const auto actionIdx : p_levinTree->actionList) {
            py::object nextState = p_applyActionFn(thisNodePtr->state, actionIdx);
            if (p_isSolutionFn(nextState)) {
                return nextState;
            }
            // Generate and push next node
            const Node* nextNodePtr = &p_levinTree->nodePool[++p_levinTree->nodePoolSize];
            nextNodePtr->state          = nextState;
            nextNodePtr->depth          = thisNodePtr->depth + 1.0f;
            nextNodePtr->logProbability = std::logf(/* TODO: from policy list */) + thisNodePtr->logProbability;
            nextNodePtr->cost           = std::logf(nextNodePtr->depth) - nextNodePtr->logProbability;
            nextNodePtr->parent         = thisNodePtr;
            levinTreeOpenListPush(p_levinTree, nextNodePtr);
        }
    }
}
