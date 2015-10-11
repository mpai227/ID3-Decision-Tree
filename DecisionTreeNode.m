classdef DecisionTreeNode < handle
    
    properties
        decision_attrib;        % the index of the attrib where split occurs
        available_attribs;      % the indices of available attributes
        decision;               % the class for all input data, if no split occours
        left_node;              % handle to the left leaf
        right_node;             % handle to the right leaf
        parent_node;            % handle to the parent leaf
    end
    methods
        function this = DecisionTreeNode()
            this.decision_attrib = -1;
            this.decision = -1;
            this.parent_node = [];
            this.available_attribs = [];
        end
        function find_decision_attrib(this,attrib,class)
            
            h = DecisionTreeNode.entropy_of_class(class);
            info_gain = zeros(size(this.available_attribs));
            instances = size(attrib, 1);
            no_attribs = size(attrib,2);
            
            for j = 1:no_attribs
                x0_zeros = 0;
                x0_ones = 0;
                x1_zeros = 0;
                x1_ones = 0;
                for k = 1:instances
                    
                    if (attrib(k,j) == 0)
                        if (class(k) == 0)
                            x0_zeros = x0_zeros + 1;
                        else
                            x0_ones = x0_ones + 1;
                        end
                    else
                        if (class(k) == 0)
                            x1_zeros = x1_zeros + 1;
                        else
                            x1_ones = x1_ones + 1;
                        end
                    end
                end
                
                p_ones = sum(attrib(:,j) == 1) / instances;
                p_zeros = sum(attrib(:,j) == 0) / instances;
                if p_zeros == 0
                    class_entropy_zeros = 0;
                else
                    class_entropy_zeros = p_zeros * DecisionTreeNode.entropy([x0_zeros/(x0_zeros+x0_ones) x0_ones/(x0_zeros+x0_ones)]);  
                end
                if p_ones == 0
                    class_entropy_ones = 0;
                else
                    class_entropy_ones = p_ones * DecisionTreeNode.entropy([x1_zeros/(x1_zeros+x1_ones) x1_ones/(x1_zeros+x1_ones)]);
                end
                info_gain(j) = h - (class_entropy_zeros + class_entropy_ones);
            end
            
            [max_ig, idx] = max(info_gain);
            this.decision_attrib = idx;          
        end
        
        function train(this,attrib,class)
            if (all(class,1))
                this.decision = 1;
                return
            elseif (norm(class - zeros(size(class,1),1)) == 0)
                this.decision = 0;
                return
            end
            if (size(this.available_attribs) == 0)
                this.decision = mode(class);
                return
            end               
            this.find_decision_attrib(attrib, class);
            this.left_node = DecisionTreeNode();
            this.right_node = DecisionTreeNode();
            this.left_node.available_attribs = this.available_attribs(this.available_attribs ~= this.decision_attrib);
            this.right_node.available_attribs = this.available_attribs(this.available_attribs ~= this.decision_attrib);
            ones_index = find(attrib(:, this.decision_attrib) == 1);
            zeros_index = find(attrib(:, this.decision_attrib) == 0);
            attrib(:, this.decision_attrib) = [];
            attrib_split_ones = attrib(ones_index, :);
            attrib_split_zeros = attrib(zeros_index, :);
            class_split_ones = class(ones_index);
            class_split_zeros = class(zeros_index);
            this.left_node.train(attrib_split_zeros, class_split_zeros);
            this.right_node.train(attrib_split_ones, class_split_ones);           
        end
        
        function class = classify(this,attrib)
        
            class = -ones(size(attrib,1),1); %initialize class labels to -1

            if this.decision == 0
                class = zeros(size(attrib,1), 1);
                return
            end   
            if this.decision == 1
                class = ones(size(attrib,1), 1);
                return
            end
            ones_index = find(attrib(:, this.decision_attrib) == 1);
            zeros_index = find(attrib(:, this.decision_attrib) == 0);
            attrib_split_ones = attrib(ones_index, :);
            attrib_split_zeros = attrib(zeros_index, :);            
            class(ones_index, 1) = this.left_node.classify(attrib_split_ones);
            class(zeros_index, 1) = this.right_node.classify(attrib_split_zeros);         
        end
    end
    
    methods (Static)
        
        function h = entropy(p)
            size_p = size(p,2);
            entropy = 0;
            for i=1:size_p
                if p(1,i) == 0
                    entropy = entropy;
                else
                    entropy = entropy - (p(1,i)*log2(p(1,i)));
                end
            end
            h = entropy;
        end
        
        function h = entropy_of_class(class)
            size_class = size(class,1);
            joy_count = 0;
            despair_count = 0;
            for i=1:size_class
                if (class(i,1) == 0)
                    joy_count = joy_count + 1;
                else
                    despair_count = despair_count + 1;
                end
            end
            p_joy = joy_count / (despair_count + joy_count);
            p_despair = despair_count / (despair_count + joy_count);
            
            if p_joy == 0
                class_entropy = - p_despair*log2(p_despair);
            elseif p_despair == 0
                class_entropy = - p_joy*log2(p_joy);
            else
                class_entropy = - p_joy*log2(p_joy) - p_despair*log2(p_despair);
            end
                
            h = class_entropy;
            
        end
    end
end
