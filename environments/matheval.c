// building a comman line interface in C.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define MAX_STACK 256
#define MAX_TOKEN 32

typedef enum {
    TOKEN_NUMBER,
    TOKEN_OPERATOR,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_END
} TokenType;

typedef struct {
    TokenType type;
    double value;
    char op;
} Token;

typedef struct {
    double data[MAX_STACK];
    int top;
} NumberStack;

typedef struct {
    char data[MAX_STACK];
    int top;
} OperatorStack;

// Stack operations
void push_number(NumberStack *stack, double value) {
    if (stack->top < MAX_STACK - 1) {
        stack->data[++stack->top] = value;
    }
}

double pop_number(NumberStack *stack) {
    if (stack->top >= 0) {
        return stack->data[stack->top--];
    }
    return 0.0;
}

void push_operator(OperatorStack *stack, char op) {
    if (stack->top < MAX_STACK - 1) {
        stack->data[++stack->top] = op;
    }
}

char pop_operator(OperatorStack *stack) {
    if (stack->top >= 0) {
        return stack->data[stack->top--];
    }
    return '\0';
}

char peek_operator(OperatorStack *stack) {
    if (stack->top >= 0) {
        return stack->data[stack->top];
    }
    return '\0';
}

// Get operator precedence
int get_precedence(char op) {
    switch (op) {
        case '+':
        case '-':
            return 1;
        case '*':
        case '/':
        case '%':
            return 2;
        case '^':
            return 3;
        default:
            return 0;
    }
}

// Check if operator is right associative
int is_right_associative(char op) {
    return op == '^';
}

// Tokenize input string
Token get_next_token(const char **input) {
    Token token = {TOKEN_END, 0.0, '\0'};
    
    // Skip whitespace
    while (**input && isspace(**input)) {
        (*input)++;
    }
    
    if (**input == '\0') {
        return token;
    }
    
    // Parse number
    if (isdigit(**input) || **input == '.') {
        token.type = TOKEN_NUMBER;
        char *endptr;
        token.value = strtod(*input, &endptr);
        *input = endptr;
        return token;
    }
    
    // Parse operators
    if (strchr("+-*/%^", **input)) {
        token.type = TOKEN_OPERATOR;
        token.op = **input;
        (*input)++;
        return token;
    }
    
    // Parse parentheses
    if (**input == '(') {
        token.type = TOKEN_LPAREN;
        (*input)++;
        return token;
    }
    
    if (**input == ')') {
        token.type = TOKEN_RPAREN;
        (*input)++;
        return token;
    }
    
    // Skip unknown character
    (*input)++;
    return get_next_token(input);
}

// Apply operator to two operands
double apply_operator(double left, double right, char op) {
    switch (op) {
        case '+': return left + right;
        case '-': return left - right;
        case '*': return left * right;
        case '/': 
            if (right == 0.0) {
                printf("Error: Division by zero\n");
                return 0.0;
            }
            return left / right;
        case '%': return fmod(left, right);
        case '^': return pow(left, right);
        default: return 0.0;
    }
}


// Evaluate expression using Shunting Yard algorithm
double evaluate_expression(const char *expression) {
    NumberStack numbers = {{0}, -1};
    OperatorStack operators = {{0}, -1};
    const char *input = expression;
    Token token;
    
    while ((token = get_next_token(&input)).type != TOKEN_END) {
        switch (token.type) {
            case TOKEN_NUMBER:
                push_number(&numbers, token.value);
                break;
                
            case TOKEN_OPERATOR:
                while (operators.top >= 0 && 
                       peek_operator(&operators) != '(' &&
                       (get_precedence(peek_operator(&operators)) > get_precedence(token.op) ||
                        (get_precedence(peek_operator(&operators)) == get_precedence(token.op) && 
                         !is_right_associative(token.op)))) {
                    
                    char op = pop_operator(&operators);
                    double right = pop_number(&numbers);
                    double left = pop_number(&numbers);
                    push_number(&numbers, apply_operator(left, right, op));
                }
                push_operator(&operators, token.op);
                break;
                
            case TOKEN_LPAREN:
                push_operator(&operators, '(');
                break;
                
            case TOKEN_RPAREN:
                while (operators.top >= 0 && peek_operator(&operators) != '(') {
                    char op = pop_operator(&operators);
                    double right = pop_number(&numbers);
                    double left = pop_number(&numbers);
                    push_number(&numbers, apply_operator(left, right, op));
                }
                if (operators.top >= 0) {
                    pop_operator(&operators); // Remove '('
                }
                break;
                
            default:
                break;
        }
    }
    
    // Process remaining operators
    while (operators.top >= 0) {
        char op = pop_operator(&operators);
        if (op != '(') {
            double right = pop_number(&numbers);
            double left = pop_number(&numbers);
            push_number(&numbers, apply_operator(left, right, op));
        }
    }
    
    return numbers.top >= 0 ? numbers.data[numbers.top] : 0.0;
}

int main(int argc, char *argv[]) {
    char expression[256];
    
    if (argc > 1) {
        // Use command line argument
        strcpy(expression, argv[1]);
        for (int i = 2; i < argc; i++) {
            strcat(expression, " ");
            strcat(expression, argv[i]);
        }
    } else {
        // Interactive mode
        printf("Math Expression Evaluator\n");
        printf("Supported operators: +, -, *, /, %%, ^\n");
        printf("Enter 'quit' to exit\n\n");
        
        while (1) {
            printf(">>> ");
            if (!fgets(expression, sizeof(expression), stdin)) {
                break;
            }
            
            // Remove newline
            expression[strcspn(expression, "\n")] = '\0';
            
            if (strcmp(expression, "quit") == 0) {
                break;
            }
            
            if (strlen(expression) == 0) {
                continue;
            }
            
            double result = evaluate_expression(expression);
            printf("Result: %.10g\n\n");
        }
        
        return 0;
    }
    
    // Evaluate single expression
    double result = evaluate_expression(expression);
    printf("%.10g\n", result);
    
    return 0;
}