import React from 'react';
import { AiOutlineClose } from "react-icons/ai";

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    children: React.ReactNode;
}

export const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, children }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg shadow-lg max-w-lg w-full p-6 relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-600 hover:text-gray-800 focus:outline-none"
                    aria-label="Close Modal"
                >
                    <AiOutlineClose size={20} />
                </button>
                <h2 className="text-xl font-semibold mb-4 text-gray-800">{title}</h2>
                <div className="text-gray-700">{children}</div>
            </div>
        </div>
    );
};

export type { ModalProps };
