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
};

class LevinTree
{
public:
    LevinTree(py::object p_model, py::object p_actionList, py::object p_initialState)
        : model(p_model),
          actionList(p_actionList)
          nodePool{{p_initialState, 1.0f, 0.0f, 0.0f, nullptr}}
          nodePoolSize(1)
          openList{&this->nodePool[0]}
          openListSize(1)
    {

    }

    ~LevinTree()
    {

    }

    py::object Fit(py::funtion getPolicy, py::function applyAction, py::function isSolution)
    {
        while (!this->heapIsEmpty()) {
            Node* node = this->heapTop();
            this->heapPop(node);
            for (const auto i : this->actionList) {
                py::object nextState = applyAction(node->state, i);
                if (isSolution(nextState)) {
                    return nextState;
                }
                float probability = getPolicy(nextState, i);
                ++this->nodePoolSize;
                this->nodePool[this->nodePoolSize] = {
                    .state          = nextState,
                    .depth          = node->depth + 1.0f,
                    .logProbability = std::logf(probability) + node->logProbability,
                    .parent         = node
                };
                this->nodePool[this->nodePoolSize]->cost = std::logf(this->nodePool[this->nodePoolSize]->depth)
                                                         - this->nodePool[this->nodePoolSize]->logProbability;
                this->heapPush(&this->nodePool[this->nodePoolSize]);
            }
        }
    }

private:
    void heapPush(const Node* p_node)
    {
        this->openList[++this->openListSize] = p_node;
        uint32_t i = this->openListSize;
        while (this->openList[j]->cost < p_node->cost && i > 1) {
            uint32_t j = i >> 1;
            this->openList[i] = this->openList[j];
            i = j;
        }
        this->openList[i] = p_node;
    }

    void heapPop(const Node* p_node)
    {
        Node* node = this->openList[this->openListSize--];
        uint32_t i = this->openListSize >> 1; // indices of last non-leaf node in the heap
        uint32_t j = p_node - this->openList; // index of the popped node in the heap
        while (j <= i) {
            uint32_t l = j << 1;
            uint32_t r = l | 1;
            float maxCost;
            if (this->openList[l]->cost >= this->openList[r]->cost) {
                maxCost = this->openList[l]->cost;
            }
            else {
                maxCost = this->openList[r]->cost;
                l = r;
            }
            if (this->openList[l]->cost > node->cost) {
                this->openList[j] = this->openList[l];
                j = l;
            }
            else break;
        }
        this->openList[j] = node;
    }

    const Node* heapTop()
    {
        return this->openList[1];
    }

    bool heapIsEmpty()
    {
        return this->openListSize == 0;
    }

    static constexpr uint32_t MAX_TREE_SIZE = 10000;

    py::object model;
    py::object actionList;
    Node       nodePool[MAX_TREE_SIZE + 1];
    uint32_t   nodePoolSize;
    Node*      openList[MAX_TREE_SIZE + 1];
    uint32_t   openListSize;
};
